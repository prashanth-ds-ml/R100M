from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOK_DIR = ROOT / "data" / "rachana_tokens"
DEFAULT_MODEL_PATH = ROOT / "tokenizer" / "rachana_bpe32k.model"
DEFAULT_SCAN_WINDOW = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode a few first and random EOS-bounded samples from tokens.bin.")
    p.add_argument("--tok_dir", type=Path, default=DEFAULT_TOK_DIR, help="Directory containing tokens.bin and meta.json")
    p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="SentencePiece model path")
    p.add_argument("--first_docs", type=int, default=5, help="How many first EOS-bounded documents to print")
    p.add_argument("--random_docs", type=int, default=5, help="How many random EOS-bounded samples to print")
    p.add_argument("--seed", type=int, default=42, help="Random seed for random sampling")
    p.add_argument("--max_decode_tokens", type=int, default=512, help="Max tokens to decode per sampled document")
    p.add_argument("--excerpt_chars", type=int, default=1200, help="Max decoded characters to print per sample")
    p.add_argument("--scan_window", type=int, default=DEFAULT_SCAN_WINDOW, help="Tokens to scan backward/forward when finding EOS boundaries")
    return p.parse_args()


def load_meta(tok_dir: Path) -> dict:
    meta_path = tok_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def decode_span(
    sp: spm.SentencePieceProcessor,
    tokens: np.memmap,
    span: Tuple[int, int],
    max_decode_tokens: int,
) -> Tuple[List[int], str]:
    start, end = span
    ids = tokens[start : min(end, start + max_decode_tokens)].tolist()
    if ids and ids[-1] == sp.eos_id():
        ids = ids[:-1]
    text = sp.decode(ids)
    return ids, text


def print_sample(
    label: str,
    sample_idx: int,
    span: Tuple[int, int],
    ids: List[int],
    text: str,
    excerpt_chars: int,
) -> None:
    excerpt = text[:excerpt_chars] + ("..." if len(text) > excerpt_chars else "")
    print(f"\n=== {label} {sample_idx} ===")
    print(f"token_span: {span[0]}:{span[1]} | decoded_tokens: {len(ids)} | decoded_chars: {len(text)}")
    print(excerpt)


def first_doc_spans(tokens: np.memmap, eos_id: int, count: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    idx = 0
    n_tokens = len(tokens)

    while idx < n_tokens and len(spans) < count:
        if int(tokens[idx]) == eos_id:
            spans.append((start, idx + 1))
            start = idx + 1
        idx += 1

    if start < n_tokens and len(spans) < count:
        spans.append((start, n_tokens))

    return spans[:count]


def find_prev_eos(tokens: np.memmap, pos: int, eos_id: int, scan_window: int) -> int:
    start = max(0, pos - scan_window)
    chunk = np.asarray(tokens[start:pos], dtype=np.uint32)
    if chunk.size == 0:
        return 0
    hits = np.where(chunk == eos_id)[0]
    if hits.size == 0:
        return start
    return start + int(hits[-1]) + 1


def find_next_eos(tokens: np.memmap, pos: int, eos_id: int, scan_window: int, n_tokens: int) -> int:
    end = min(n_tokens, pos + scan_window)
    chunk = np.asarray(tokens[pos:end], dtype=np.uint32)
    if chunk.size == 0:
        return n_tokens
    hits = np.where(chunk == eos_id)[0]
    if hits.size == 0:
        return end
    return pos + int(hits[0]) + 1


def random_doc_spans(
    tokens: np.memmap,
    eos_id: int,
    count: int,
    seed: int,
    scan_window: int,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    n_tokens = len(tokens)
    spans: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()

    attempts = 0
    max_attempts = max(50, count * 20)
    while len(spans) < count and attempts < max_attempts:
        attempts += 1
        pos = rng.randrange(0, n_tokens)
        start = find_prev_eos(tokens, pos, eos_id, scan_window)
        end = find_next_eos(tokens, pos, eos_id, scan_window, n_tokens)
        span = (start, end)
        if end <= start or span in seen:
            continue
        seen.add(span)
        spans.append(span)

    spans.sort()
    return spans


def main() -> None:
    args = parse_args()
    meta = load_meta(args.tok_dir)

    tok_path = args.tok_dir / "tokens.bin"
    eos_id = int(meta["eos_id"])
    n_tokens = int(meta["n_tokens"])

    tokens = np.memmap(tok_path, dtype=np.uint32, mode="r", shape=(n_tokens,))

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.model_path))

    print("Loaded:", tok_path)
    print("n_tokens:", n_tokens)
    print("eos_id:", eos_id)

    first_spans = first_doc_spans(tokens, eos_id, args.first_docs)
    for idx, span in enumerate(first_spans, start=1):
        ids, text = decode_span(sp, tokens, span, args.max_decode_tokens)
        print_sample("FIRST", idx, span, ids, text, args.excerpt_chars)

    random_spans = random_doc_spans(tokens, eos_id, args.random_docs, args.seed, args.scan_window)
    for idx, span in enumerate(random_spans, start=1):
        ids, text = decode_span(sp, tokens, span, args.max_decode_tokens)
        print_sample("RANDOM", idx, span, ids, text, args.excerpt_chars)


if __name__ == "__main__":
    main()
