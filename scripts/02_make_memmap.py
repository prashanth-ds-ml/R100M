from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "rachana_tokens"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open tokens.bin as a memmap and print a quick sanity summary.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory containing tokens.bin and meta.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.out_dir / "meta.json"
    bin_path = args.out_dir / "tokens.bin"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    dtype = np.uint32
    n_tokens = int(meta["n_tokens"])

    tokens = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_tokens,))

    print("Loaded memmap:", bin_path)
    print("n_tokens:", n_tokens)
    print("dtype:", dtype.__name__)
    print("First 20:", tokens[:20].tolist())
    print("Last 20:", tokens[-20:].tolist())


if __name__ == "__main__":
    main()
