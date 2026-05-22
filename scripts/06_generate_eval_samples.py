import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import sentencepiece as spm
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path: Path) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            prompt_id = row.get("id")
            category = row.get("category")
            prompt = row.get("prompt")
            if not prompt_id or not category or not prompt:
                raise ValueError(f"Invalid prompt row at line {line_no}: expected id/category/prompt")
            prompts.append(
                {
                    "id": str(prompt_id),
                    "category": str(category),
                    "prompt": str(prompt),
                }
            )
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


@torch.no_grad()
def generate_text(
    model,
    sp,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
    device: str,
) -> List[int]:
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids=x).logits[:, -1, :]
        logits = logits / max(temperature, 1e-8)

        if top_k and top_k > 0:
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            sampled = indices[0, torch.multinomial(probs[0], 1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs[0], 1)

        next_id = int(sampled.item())
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == eos_id:
            break

    return x[0].tolist()


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, payload: Dict[str, Any]):
    lines: List[str] = []
    lines.append(f"# Generation Eval - {payload['run_name']} - {payload['checkpoint_name']}")
    lines.append("")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(f"- temperatures: `{payload['temperatures']}`")
    lines.append(f"- top_k: `{payload['top_k']}`")
    lines.append(f"- max_new_tokens: `{payload['max_new_tokens']}`")
    lines.append(f"- prompt_file: `{payload['prompt_file']}`")
    lines.append("")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in payload["results"]:
        grouped.setdefault(row["prompt_category"], []).append(row)

    for category in sorted(grouped):
        lines.append(f"## {category}")
        lines.append("")
        category_rows = grouped[category]
        prompt_ids = []
        seen = set()
        for row in category_rows:
            if row["prompt_id"] not in seen:
                seen.add(row["prompt_id"])
                prompt_ids.append(row["prompt_id"])

        for prompt_id in prompt_ids:
            sample_rows = [row for row in category_rows if row["prompt_id"] == prompt_id]
            base = sample_rows[0]
            lines.append(f"### {prompt_id}")
            lines.append("")
            lines.append(f"**Prompt**: {base['prompt']}")
            lines.append("")
            for row in sample_rows:
                lines.append(f"#### temperature={row['temperature']}")
                lines.append("")
                lines.append(row["generated_text"])
                lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate standardized qualitative samples from a trained run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--tok_dir", type=Path, default=Path("data") / "rachana_tokens")
    parser.add_argument("--model_path", type=Path, default=Path("tokenizer") / "rachana_bpe32k.model")
    parser.add_argument("--prompt_file", type=Path, default=Path("eval_prompts") / "generation_prompts.jsonl")
    parser.add_argument("--checkpoint", choices=["latest", "best", "both"], default="both")
    parser.add_argument("--temperatures", type=str, default="0.7,0.9,1.1")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.run_dir.resolve()
    tok_dir = args.tok_dir.resolve()
    sp_model_path = args.model_path.resolve()
    prompt_file = args.prompt_file.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    tok_meta = load_json(tok_dir / "meta.json")
    eos_id = int(tok_meta["eos_id"])
    prompts = load_prompts(prompt_file)
    temperatures = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    ckpt_names = ["latest", "best"] if args.checkpoint == "both" else [args.checkpoint]
    out_dir = run_dir / "generation_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_name in ckpt_names:
        ckpt_dir = run_dir / "checkpoints" / ckpt_name
        if not ckpt_dir.exists():
            print(f"[skip] checkpoint not found: {ckpt_dir}")
            continue

        print(f"Loading checkpoint: {ckpt_dir}")
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir)).to(device)

        results: List[Dict[str, Any]] = []
        total = len(prompts) * len(temperatures)
        with tqdm(total=total, desc=f"Generate {ckpt_name}", dynamic_ncols=True) as pbar:
            for prompt_row in prompts:
                prompt = prompt_row["prompt"]
                prompt_ids = sp.encode(prompt, out_type=int)

                for temperature in temperatures:
                    full_ids = generate_text(
                        model=model,
                        sp=sp,
                        prompt_ids=prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temperature,
                        top_k=args.top_k,
                        eos_id=eos_id,
                        device=device,
                    )
                    full_text = sp.decode(full_ids)
                    continuation = full_text[len(prompt):].lstrip() if full_text.startswith(prompt) else full_text
                    results.append(
                        {
                            "run_name": run_dir.name,
                            "checkpoint_name": ckpt_name,
                            "prompt_id": prompt_row["id"],
                            "prompt_category": prompt_row["category"],
                            "temperature": temperature,
                            "top_k": args.top_k,
                            "max_new_tokens": args.max_new_tokens,
                            "prompt": prompt,
                            "generated_text": full_text,
                            "continuation_text": continuation,
                            "prompt_token_count": len(prompt_ids),
                            "generated_token_count": len(full_ids),
                        }
                    )
                    pbar.update(1)

        payload = {
            "run_name": run_dir.name,
            "checkpoint_name": ckpt_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "prompt_file": str(prompt_file),
            "temperatures": temperatures,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "result_count": len(results),
            "results": results,
        }

        json_out = out_dir / f"{ckpt_name}_generations.json"
        md_out = out_dir / f"{ckpt_name}_generations.md"
        write_json(json_out, payload)
        write_markdown(md_out, payload)
        print(f"Saved JSON: {json_out}")
        print(f"Saved Markdown: {md_out}")


if __name__ == "__main__":
    main()
