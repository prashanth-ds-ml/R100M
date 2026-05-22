import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOK_DIR = REPO_ROOT / "data" / "rachana_tokens"
DEFAULT_SP_MODEL = REPO_ROOT / "tokenizer" / "rachana_bpe32k.model"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs"


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    scale: str
    trainer: str
    run_name: str
    args: Dict[str, Any]


COMMON_PRESETS = {
    "tiny": {
        "max_length": 512,
        "stride": 512,
        "batch_size": 1,
        "grad_accum": 16,
        "max_steps": 15000,
        "save_every_updates": 100,
        "eval_every_updates": 100,
        "val_fraction": 0.005,
        "grad_ckpt": True,
    },
    "small": {
        "max_length": 768,
        "stride": 768,
        "batch_size": 1,
        "grad_accum": 32,
        "max_steps": 15000,
        "save_every_updates": 200,
        "eval_every_updates": 200,
        "val_fraction": 0.005,
        "grad_ckpt": True,
    },
    "base": {
        "max_length": 1024,
        "stride": 1024,
        "batch_size": 1,
        "grad_accum": 32,
        "max_steps": 15000,
        "save_every_updates": 250,
        "eval_every_updates": 250,
        "val_fraction": 0.005,
        "grad_ckpt": True,
    },
}


MODEL_PRESETS = {
    "tiny": {
        "gpt": {"n_layer": 8, "n_head": 8, "n_embd": 512},
        "llama": {"n_layer": 8, "n_head": 8, "n_embd": 512, "n_kv_head": 4, "intermediate_size": 1536},
        "mistral": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "n_kv_head": 4,
            "intermediate_size": 1536,
            "sliding_window": 512,
        },
        "hybrid_llama": {
            "backbone": "llama",
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "n_kv_head": 4,
            "intermediate_size": 1536,
        },
        "hybrid_mistral": {
            "backbone": "mistral",
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "n_kv_head": 4,
            "intermediate_size": 1536,
            "sliding_window": 512,
        },
    },
    "small": {
        "gpt": {"n_layer": 12, "n_head": 12, "n_embd": 768},
        "llama": {"n_layer": 12, "n_head": 12, "n_embd": 768, "n_kv_head": 4, "intermediate_size": 2304},
        "mistral": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_head": 4,
            "intermediate_size": 2304,
            "sliding_window": 768,
        },
        "hybrid_llama": {
            "backbone": "llama",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_head": 4,
            "intermediate_size": 2304,
        },
        "hybrid_mistral": {
            "backbone": "mistral",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_head": 4,
            "intermediate_size": 2304,
            "sliding_window": 768,
        },
    },
    "base": {
        "gpt": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
        "llama": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "n_kv_head": 8, "intermediate_size": 3072},
        "mistral": {
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "n_kv_head": 8,
            "intermediate_size": 3072,
            "sliding_window": 1024,
        },
        "hybrid_llama": {
            "backbone": "llama",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "n_kv_head": 8,
            "intermediate_size": 3072,
        },
        "hybrid_mistral": {
            "backbone": "mistral",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "n_kv_head": 8,
            "intermediate_size": 3072,
            "sliding_window": 1024,
        },
    },
}


TRAINER_BY_FAMILY = {
    "gpt": "04_train_hf_gpt2.py",
    "llama": "04_train_hf_llama.py",
    "mistral": "04_train_hf_mistral.py",
    "hybrid_llama": "04_train_hf_hybrid.py",
    "hybrid_mistral": "04_train_hf_hybrid.py",
}


def resolve_run_name(family: str, scale: str) -> str:
    if family.startswith("hybrid_"):
        return f"rachana_{family}_{scale}"
    return f"rachana_{family}_{scale}"


def build_specs(families: List[str], scales: List[str], base_args: Dict[str, Any]) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    for scale in scales:
        for family in families:
            merged = dict(COMMON_PRESETS[scale])
            merged.update(MODEL_PRESETS[scale][family])
            merged.update(base_args)
            run_name = resolve_run_name(family, scale)
            specs.append(
                ExperimentSpec(
                    family=family,
                    scale=scale,
                    trainer=TRAINER_BY_FAMILY[family],
                    run_name=run_name,
                    args=merged,
                )
            )
    return specs


def maybe_increment_max_steps(spec: ExperimentSpec, run_root: Path, step_increment: int) -> ExperimentSpec:
    run_dir = run_root / spec.run_name
    cfg_snapshot_path = run_dir / "cfg_snapshot.json"
    trainer_state_path = run_dir / "checkpoints" / "latest" / "trainer_state.pt"

    args = dict(spec.args)
    current_target = int(args["max_steps"])
    previous_target = current_target
    trainer_target = current_target
    micro_step = 0

    if not cfg_snapshot_path.exists():
        previous_target = current_target
    else:
        try:
            with cfg_snapshot_path.open("r", encoding="utf-8") as f:
                prev_cfg = json.load(f)
            previous_target = int(prev_cfg.get("max_steps", current_target))
            current_target = max(current_target, previous_target)
        except Exception:
            previous_target = current_target

    if trainer_state_path.exists():
        try:
            import torch

            state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            micro_step = int(state.get("micro_step", 0))
            trainer_target = int(state.get("cfg_snapshot", {}).get("max_steps", current_target))
            current_target = max(current_target, trainer_target)
        except Exception:
            micro_step = 0

    if micro_step >= current_target:
        args["max_steps"] = current_target + step_increment
    else:
        args["max_steps"] = current_target

    return ExperimentSpec(spec.family, spec.scale, spec.trainer, spec.run_name, args)


def refresh_cfg_snapshot(spec: ExperimentSpec, tok_dir: Path, sp_model_path: Path, run_root: Path):
    run_dir = run_root / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_snapshot = {
        "tok_dir": str(tok_dir),
        "sp_model_path": str(sp_model_path),
        "run_dir": str(run_dir),
        **spec.args,
    }
    with (run_dir / "cfg_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, ensure_ascii=False, indent=2)


def to_cli_args(spec: ExperimentSpec, tok_dir: Path, sp_model_path: Path, run_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / spec.trainer),
        "--tok_dir",
        str(tok_dir),
        "--sp_model_path",
        str(sp_model_path),
        "--run_dir",
        str(run_root / spec.run_name),
    ]
    for key, value in spec.args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def estimate_params(spec: ExperimentSpec, vocab_size: int = 32000) -> int:
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )

    args = spec.args
    max_length = int(args["max_length"])
    if spec.family == "gpt":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_positions=max_length,
                n_ctx=max_length,
                n_layer=int(args["n_layer"]),
                n_head=int(args["n_head"]),
                n_embd=int(args["n_embd"]),
                bos_token_id=1,
                eos_token_id=3,
            )
        )
    elif spec.family == "llama":
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=int(args["n_embd"]),
                intermediate_size=int(args["intermediate_size"]),
                num_hidden_layers=int(args["n_layer"]),
                num_attention_heads=int(args["n_head"]),
                num_key_value_heads=int(args["n_kv_head"]),
                max_position_embeddings=max_length,
                bos_token_id=1,
                eos_token_id=3,
                pad_token_id=0,
            )
        )
    elif spec.family in {"mistral", "hybrid_mistral"}:
        model = MistralForCausalLM(
            MistralConfig(
                vocab_size=vocab_size,
                hidden_size=int(args["n_embd"]),
                intermediate_size=int(args["intermediate_size"]),
                num_hidden_layers=int(args["n_layer"]),
                num_attention_heads=int(args["n_head"]),
                num_key_value_heads=int(args["n_kv_head"]),
                max_position_embeddings=max_length,
                sliding_window=int(args["sliding_window"]),
                bos_token_id=1,
                eos_token_id=3,
                pad_token_id=0,
            )
        )
    elif spec.family == "hybrid_llama":
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=int(args["n_embd"]),
                intermediate_size=int(args["intermediate_size"]),
                num_hidden_layers=int(args["n_layer"]),
                num_attention_heads=int(args["n_head"]),
                num_key_value_heads=int(args["n_kv_head"]),
                max_position_embeddings=max_length,
                bos_token_id=1,
                eos_token_id=3,
                pad_token_id=0,
            )
        )
    else:
        raise ValueError(f"Unsupported family for param estimate: {spec.family}")
    return sum(p.numel() for p in model.parameters())


def print_plan(specs: List[ExperimentSpec], tok_dir: Path, sp_model_path: Path, run_root: Path, show_params: bool):
    print("\n=== Experiment Queue ===")
    for idx, spec in enumerate(specs, start=1):
        params = ""
        if show_params:
            n_params = estimate_params(spec)
            params = f" | params={n_params / 1e6:.2f}M"
        print(f"{idx}. {spec.run_name} | trainer={spec.trainer}{params}")
        print("   " + " ".join(to_cli_args(spec, tok_dir, sp_model_path, run_root)))


def save_manifest(specs: List[ExperimentSpec], run_root: Path):
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [
            {
                "family": spec.family,
                "scale": spec.scale,
                "trainer": spec.trainer,
                "run_name": spec.run_name,
                "args": spec.args,
            }
            for spec in specs
        ],
    }
    path = run_root / "experiment_queue_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def run_queue(specs: List[ExperimentSpec], tok_dir: Path, sp_model_path: Path, run_root: Path, continue_on_error: bool):
    for spec in specs:
        refresh_cfg_snapshot(spec, tok_dir, sp_model_path, run_root)
        cmd = to_cli_args(spec, tok_dir, sp_model_path, run_root)
        print(f"\n=== Running {spec.run_name} ===")
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as exc:
            print(f"[error] {spec.run_name} failed with exit code {exc.returncode}")
            if not continue_on_error:
                raise


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential experiment queue for Rachana model families.")
    parser.add_argument(
        "--families",
        type=str,
        default="gpt,llama,mistral,hybrid_llama,hybrid_mistral",
        help="Comma-separated families",
    )
    parser.add_argument("--scales", type=str, default="tiny", help="Comma-separated scales: tiny,small,base")
    parser.add_argument("--tok_dir", type=Path, default=DEFAULT_TOK_DIR)
    parser.add_argument("--sp_model_path", type=Path, default=DEFAULT_SP_MODEL)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every_updates", type=int, default=20)
    parser.add_argument("--sample_every_updates", type=int, default=200)
    parser.add_argument("--sample_max_new_tokens", type=int, default=120)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--show_params", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--step_increment", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    scales = [x.strip() for x in args.scales.split(",") if x.strip()]

    for scale in scales:
        if scale not in COMMON_PRESETS:
            raise ValueError(f"Unknown scale: {scale}")
    for family in families:
        if family not in TRAINER_BY_FAMILY:
            raise ValueError(f"Unknown family: {family}")

    base_args = {
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "clip_grad_norm": args.clip_grad_norm,
        "log_every_updates": args.log_every_updates,
        "sample_every_updates": args.sample_every_updates,
        "sample_max_new_tokens": args.sample_max_new_tokens,
        "sample_temperature": args.sample_temperature,
        "sample_top_k": args.sample_top_k,
        "eval_batches": args.eval_batches,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }

    specs = build_specs(families, scales, base_args)
    specs = [maybe_increment_max_steps(spec, args.run_root, args.step_increment) for spec in specs]
    manifest = save_manifest(specs, args.run_root)
    print(f"Saved queue manifest: {manifest}")
    print_plan(specs, args.tok_dir, args.sp_model_path, args.run_root, args.show_params)

    if args.dry_run:
        return

    run_queue(specs, args.tok_dir, args.sp_model_path, args.run_root, args.continue_on_error)


if __name__ == "__main__":
    main()
