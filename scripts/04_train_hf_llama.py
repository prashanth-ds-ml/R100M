# scripts/04_train_hf_llama.py
# Unified run folder + TRUE auto-resume + disk-safe (keeps ONLY latest & best)
# Mirrors the GPT trainer flow so runs/logs/checkpoints fit the same dashboard pipeline.
#
# Usage example:
#   python scripts/04_train_hf_llama.py `
#     --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
#     --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
#     --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_llama_tiny_80m" `
#     --max_length 512 --stride 512 `
#     --n_layer 8 --n_head 8 --n_embd 512 `
#     --batch_size 1 --grad_accum 16 `
#     --max_steps 5000 `
#     --grad_ckpt `
#     --save_every_updates 200 `
#     --eval_every_updates 200 `
#     --val_fraction 0.005
#
import os, json, time, csv, subprocess, random, argparse, shutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup


@dataclass
class TrainConfig:
    tok_dir: str
    sp_model_path: str
    run_dir: str

    # data
    max_length: int
    stride: int
    num_workers: int

    # model
    n_layer: int
    n_head: int
    n_embd: int
    n_kv_head: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    gradient_checkpointing: bool

    # training
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    clip_grad_norm: float

    # io
    log_every_updates: int
    save_every_updates: int
    sample_every_updates: int
    sample_max_new_tokens: int
    sample_temperature: float
    sample_top_k: int

    # eval (optional)
    eval_every_updates: int
    val_fraction: float
    eval_batches: int

    # reproducibility
    seed: int
    deterministic: bool


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(deterministic: bool):
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def ensure_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")


def init_csv(path: str):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "micro_step", "update_step",
            "train_loss", "train_ppl",
            "val_loss", "val_ppl",
            "lr", "tokens_in_update", "tokens_total",
            "tok_per_sec", "secs",
            "grad_norm", "oom_skips",
            "n_params", "tokens_per_param"
        ])


def append_csv(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_rng_state(path: str):
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(st, path)


def load_rng_state(path: str):
    st = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and st["cuda"] is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


def safe_exp(x: float, cap: float = 50.0) -> float:
    x = float(min(x, cap))
    return float(np.exp(x))


def atomic_replace_dir(src_dir: str, dst_dir: str):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.move(src_dir, dst_dir)


class RachanaStreamDataset(Dataset):
    def __init__(self, token_ids_memmap, max_length: int, stride: int, start_offset: int = 0, end_offset: Optional[int] = None):
        self.tokens = token_ids_memmap
        self.max_length = max_length
        self.stride = stride

        end_offset = len(self.tokens) if end_offset is None else int(end_offset)
        start_offset = int(start_offset)
        if end_offset <= start_offset:
            raise ValueError("Invalid dataset slice: end_offset must be > start_offset")

        self.start_offset = start_offset
        self.end_offset = end_offset

        usable = self.end_offset - self.start_offset
        self.n_samples = (usable - (max_length + 1)) // stride + 1
        assert self.n_samples > 0, "Not enough tokens for given max_length/stride in selected slice."

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        i = self.start_offset + idx * self.stride
        x = self.tokens[i : i + self.max_length].astype(np.int64, copy=False)
        y = self.tokens[i + 1 : i + self.max_length + 1].astype(np.int64, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def generate_sample(model, sp, prompt_ids, max_new_tokens, temperature, top_k, eos_id, device) -> str:
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids=x).logits[:, -1, :]
        logits = logits / max(temperature, 1e-8)

        if top_k and top_k > 0:
            v, ix = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.softmax(v, dim=-1)
            next_token = ix[0, torch.multinomial(probs[0], 1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs[0], 1)

        next_id = int(next_token.item())
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break

    out = sp.decode(x[0].tolist())
    model.train()
    return out


@torch.no_grad()
def evaluate_loss(model: LlamaForCausalLM, loader: DataLoader, device: str, use_amp: bool, max_batches: int) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            out = model(input_ids=x, labels=y)
            loss = out.loss
        if torch.isfinite(loss):
            losses.append(float(loss.detach().cpu()))
    model.train()
    if len(losses) == 0:
        return float("inf")
    return float(np.mean(losses))


def save_checkpoint_latest(run_ckpt_dir: str,
                           micro_step: int,
                           update_step: int,
                           accum_count: int,
                           tokens_total: int,
                           best_val_loss: float,
                           model: LlamaForCausalLM,
                           optimizer: torch.optim.Optimizer,
                           scheduler,
                           cfg: TrainConfig,
                           extra: Dict[str, Any]):
    latest_dir = os.path.join(run_ckpt_dir, "latest")
    tmp_dir = os.path.join(run_ckpt_dir, "_tmp_latest")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    model.save_pretrained(tmp_dir)
    torch.save(
        {
            "micro_step": micro_step,
            "update_step": update_step,
            "accum_count": accum_count,
            "tokens_total": int(tokens_total),
            "best_val_loss": float(best_val_loss),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "extra": extra,
            "cfg_snapshot": asdict(cfg),
        },
        os.path.join(tmp_dir, "trainer_state.pt"),
    )
    save_rng_state(os.path.join(tmp_dir, "rng_state.pt"))

    atomic_replace_dir(tmp_dir, latest_dir)
    print(f"\n[saved] latest @ {latest_dir}")


def save_checkpoint_best(run_ckpt_dir: str,
                         micro_step: int,
                         update_step: int,
                         accum_count: int,
                         tokens_total: int,
                         best_val_loss: float,
                         model: LlamaForCausalLM,
                         optimizer: torch.optim.Optimizer,
                         scheduler,
                         cfg: TrainConfig,
                         extra: Dict[str, Any]):
    best_dir = os.path.join(run_ckpt_dir, "best")
    tmp_dir = os.path.join(run_ckpt_dir, "_tmp_best")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    model.save_pretrained(tmp_dir)
    torch.save(
        {
            "micro_step": micro_step,
            "update_step": update_step,
            "accum_count": accum_count,
            "tokens_total": int(tokens_total),
            "best_val_loss": float(best_val_loss),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "extra": extra,
            "cfg_snapshot": asdict(cfg),
        },
        os.path.join(tmp_dir, "trainer_state.pt"),
    )
    save_rng_state(os.path.join(tmp_dir, "rng_state.pt"))

    atomic_replace_dir(tmp_dir, best_dir)
    print(f"\n[saved] best @ {best_dir}")


def load_latest_if_any(run_ckpt_dir: str, device: str) -> Optional[Tuple[str, Dict[str, Any], LlamaForCausalLM]]:
    latest_dir = os.path.join(run_ckpt_dir, "latest")
    st_path = os.path.join(latest_dir, "trainer_state.pt")
    if not os.path.isfile(st_path):
        return None
    st = torch.load(st_path, map_location="cpu", weights_only=False)
    model = LlamaForCausalLM.from_pretrained(latest_dir).to(device)
    return latest_dir, st, model


def resolve_intermediate_size(n_embd: int, intermediate_size: int) -> int:
    if intermediate_size > 0:
        return intermediate_size
    # LLaMA-style default is close to 8/3 * hidden_size, rounded to a multiple of 256.
    raw = int((8 * n_embd) / 3)
    return int(((raw + 255) // 256) * 256)


def main(cfg: TrainConfig):
    os.makedirs(cfg.run_dir, exist_ok=True)
    set_seed(cfg.seed)
    set_determinism(cfg.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt_root = os.path.join(cfg.run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    csv_path = os.path.join(cfg.run_dir, "train_log.csv")
    samples_path = os.path.join(cfg.run_dir, "samples.txt")
    ensure_file(samples_path)
    init_csv(csv_path)

    sp = spm.SentencePieceProcessor()
    sp.load(cfg.sp_model_path)

    meta = load_json(os.path.join(cfg.tok_dir, "meta.json"))
    n_tokens = int(meta["n_tokens"])
    vocab_size = int(meta["vocab_size"])
    pad_id = int(meta["pad_id"])
    bos_id = int(meta["bos_id"])
    eos_id = int(meta["eos_id"])

    tokens = np.memmap(os.path.join(cfg.tok_dir, "tokens.bin"), dtype=np.uint32, mode="r", shape=(n_tokens,))
    print(f"Tokens: {n_tokens} | Vocab: {vocab_size} | EOS: {eos_id}")

    meta_path = os.path.join(cfg.run_dir, "run_metadata.json")
    if not os.path.exists(meta_path):
        save_json(meta_path, {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": try_get_git_commit(),
            "meta_json": meta,
            "train_config": asdict(cfg),
            "model_family": "llama",
        })

    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError(f"Invalid config: n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head}).")
    if cfg.n_head % cfg.n_kv_head != 0:
        raise ValueError(f"Invalid config: n_head ({cfg.n_head}) must be divisible by n_kv_head ({cfg.n_kv_head}).")

    if cfg.val_fraction and cfg.val_fraction > 0.0:
        val_tokens = int(n_tokens * cfg.val_fraction)
        val_start = max(0, n_tokens - val_tokens)
        train_start, train_end = 0, val_start
        val_start, val_end = val_start, n_tokens
        if train_end - train_start < (cfg.max_length + 2):
            raise ValueError("Train slice too small after applying val_fraction. Reduce val_fraction.")
        if val_end - val_start < (cfg.max_length + 2):
            raise ValueError("Val slice too small for max_length. Reduce max_length or increase val_fraction.")
        train_ds = RachanaStreamDataset(tokens, cfg.max_length, cfg.stride, start_offset=train_start, end_offset=train_end)
        val_ds = RachanaStreamDataset(tokens, cfg.max_length, cfg.stride, start_offset=val_start, end_offset=val_end)
    else:
        train_ds = RachanaStreamDataset(tokens, cfg.max_length, cfg.stride, start_offset=0, end_offset=n_tokens)
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
    )
    print("Train samples:", len(train_ds))

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        print("Val samples:", len(val_ds))

    start_micro_step = 0
    update_step = 0
    accum_count = 0
    tokens_total = 0
    best_val_loss = float("inf")
    oom_skips = 0

    resumed = load_latest_if_any(ckpt_root, device)
    if resumed is not None:
        latest_dir, st, model = resumed
        print("Auto-resume from:", latest_dir)

        start_micro_step = int(st.get("micro_step", 0))
        update_step = int(st.get("update_step", 0))
        accum_count = int(st.get("accum_count", 0))
        tokens_total = int(st.get("tokens_total", 0))
        best_val_loss = float(st.get("best_val_loss", float("inf")))
        extra = st.get("extra", {}) or {}
        oom_skips = int(extra.get("oom_skips", 0))

        rng_path = os.path.join(latest_dir, "rng_state.pt")
        if os.path.exists(rng_path):
            load_rng_state(rng_path)
            print("RNG state restored.")
    else:
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=cfg.n_embd,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.n_layer,
            num_attention_heads=cfg.n_head,
            num_key_value_heads=cfg.n_kv_head,
            max_position_embeddings=cfg.max_length,
            hidden_act="silu",
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            tie_word_embeddings=False,
            attention_bias=False,
            mlp_bias=False,
        )
        model = LlamaForCausalLM(llama_cfg).to(device)
        print("Starting fresh LLaMA-style run.")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps
    )

    if resumed is not None:
        st = resumed[1]
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        optimizer.zero_grad(set_to_none=True)

    if start_micro_step >= cfg.max_steps:
        print(f"Already finished: start_micro_step={start_micro_step} >= max_steps={cfg.max_steps}. Exiting.")
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    data_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, initial=start_micro_step, desc="training", dynamic_ncols=True)

    micro_step = start_micro_step
    last_val_loss = float("nan")
    last_val_ppl = float("nan")

    while micro_step < cfg.max_steps:
        t0 = time.time()

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        try:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = model(input_ids=x, labels=y)
                loss = out.loss / cfg.grad_accum

            if not torch.isfinite(loss):
                print("\n[warn] non-finite loss; skipping micro-step")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                micro_step += 1
                pbar.update(1)
                continue

            scaler.scale(loss).backward()
            accum_count += 1

            did_update = False
            grad_norm_val = None
            train_loss_true = None
            tokens_in_update = cfg.batch_size * cfg.max_length * cfg.grad_accum

            if accum_count >= cfg.grad_accum:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                grad_norm_val = float(grad_norm.detach().cpu())

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                accum_count = 0
                did_update = True
                update_step += 1
                tokens_total += tokens_in_update
                train_loss_true = float(loss.detach().cpu()) * cfg.grad_accum

            if did_update and val_loader is not None and cfg.eval_every_updates > 0 and (update_step % cfg.eval_every_updates == 0):
                last_val_loss = evaluate_loss(model, val_loader, device, use_amp, max_batches=cfg.eval_batches)
                last_val_ppl = safe_exp(last_val_loss)

                if last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    save_checkpoint_best(
                        ckpt_root,
                        micro_step, update_step, accum_count, tokens_total, best_val_loss,
                        model, optimizer, scheduler, cfg,
                        extra={"oom_skips": oom_skips, "note": "best_by_val"}
                    )

            if did_update and (update_step % cfg.log_every_updates == 0):
                lr_now = scheduler.get_last_lr()[0]
                secs = time.time() - t0
                tok_per_sec = tokens_in_update / max(secs, 1e-8)
                train_ppl = safe_exp(train_loss_true if train_loss_true is not None else 999.0)
                tokens_per_param = tokens_total / max(n_params, 1)

                pbar.set_postfix(loss=f"{train_loss_true:.4f}", lr=f"{lr_now:.2e}", oom=oom_skips)
                append_csv(csv_path, [
                    micro_step, update_step,
                    train_loss_true, train_ppl,
                    (last_val_loss if val_loader is not None else ""), (last_val_ppl if val_loader is not None else ""),
                    lr_now, tokens_in_update, tokens_total,
                    tok_per_sec, secs,
                    grad_norm_val, oom_skips,
                    n_params, tokens_per_param
                ])

            if did_update and cfg.save_every_updates > 0 and (update_step % cfg.save_every_updates == 0):
                save_checkpoint_latest(
                    ckpt_root,
                    micro_step, update_step, accum_count, tokens_total, best_val_loss,
                    model, optimizer, scheduler, cfg,
                    extra={"oom_skips": oom_skips}
                )

            if did_update and cfg.sample_every_updates > 0 and (update_step % cfg.sample_every_updates == 0):
                prompt = x[0][:32].tolist()
                decoded = generate_sample(
                    model, sp,
                    prompt_ids=prompt,
                    max_new_tokens=cfg.sample_max_new_tokens,
                    temperature=cfg.sample_temperature,
                    top_k=cfg.sample_top_k,
                    eos_id=eos_id,
                    device=device,
                )
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                txt = f"\n=== update {update_step} | micro_step {micro_step} | {stamp} ===\n{decoded}\n"
                print(txt)
                with open(samples_path, "a", encoding="utf-8") as f:
                    f.write(txt)

        except torch.cuda.OutOfMemoryError:
            oom_skips += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            micro_step += 1
            pbar.update(1)

    save_checkpoint_latest(
        ckpt_root,
        micro_step, update_step, accum_count, tokens_total, best_val_loss,
        model, optimizer, scheduler, cfg,
        extra={"oom_skips": oom_skips, "final": True}
    )
    pbar.close()
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--tok_dir", type=str, required=True)
    p.add_argument("--sp_model_path", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True, help="Unified run folder (auto-resume inside it)")

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--n_kv_head", type=int, default=None, help="Defaults to n_head when omitted")
    p.add_argument("--intermediate_size", type=int, default=0, help="0 => use LLaMA-style derived default")
    p.add_argument("--rms_norm_eps", type=float, default=1e-6)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--grad_ckpt", action="store_true")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument("--log_every_updates", type=int, default=20)
    p.add_argument("--save_every_updates", type=int, default=200)
    p.add_argument("--sample_every_updates", type=int, default=200)
    p.add_argument("--sample_max_new_tokens", type=int, default=120)
    p.add_argument("--sample_temperature", type=float, default=1.0)
    p.add_argument("--sample_top_k", type=int, default=50)

    p.add_argument("--eval_every_updates", type=int, default=200, help="0 disables eval")
    p.add_argument("--val_fraction", type=float, default=0.005, help="0 disables val split")
    p.add_argument("--eval_batches", type=int, default=50, help="cap eval cost")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    n_kv_head = a.n_head if a.n_kv_head is None else a.n_kv_head
    intermediate_size = resolve_intermediate_size(a.n_embd, a.intermediate_size)

    cfg = TrainConfig(
        tok_dir=a.tok_dir,
        sp_model_path=a.sp_model_path,
        run_dir=a.run_dir,

        max_length=a.max_length,
        stride=a.stride,
        num_workers=a.num_workers,

        n_layer=a.n_layer,
        n_head=a.n_head,
        n_embd=a.n_embd,
        n_kv_head=n_kv_head,
        intermediate_size=intermediate_size,
        rms_norm_eps=a.rms_norm_eps,
        rope_theta=a.rope_theta,
        gradient_checkpointing=a.grad_ckpt,

        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        lr=a.lr,
        weight_decay=a.weight_decay,
        warmup_steps=a.warmup_steps,
        max_steps=a.max_steps,
        clip_grad_norm=a.clip_grad_norm,

        log_every_updates=a.log_every_updates,
        save_every_updates=a.save_every_updates,
        sample_every_updates=a.sample_every_updates,
        sample_max_new_tokens=a.sample_max_new_tokens,
        sample_temperature=a.sample_temperature,
        sample_top_k=a.sample_top_k,

        eval_every_updates=a.eval_every_updates,
        val_fraction=a.val_fraction,
        eval_batches=a.eval_batches,

        seed=a.seed,
        deterministic=a.deterministic,
    )

    os.makedirs(cfg.run_dir, exist_ok=True)
    cfg_snap = os.path.join(cfg.run_dir, "cfg_snapshot.json")
    if not os.path.exists(cfg_snap):
        save_json(cfg_snap, asdict(cfg))

    main(cfg)
