# Training Guide

This project now supports two training styles:
- sequential automation with the queue runner
- manual one-model-at-a-time training

For the current workflow, manual one-model-at-a-time training is recommended because:
- long runs are easier to control
- you can choose exactly which family to continue next
- it avoids waiting for unrelated models in the queue

## Token Source

All current trainers use:
- `data\rachana_tokens\tokens.bin`
- `data\rachana_tokens\meta.json`
- `tokenizer\rachana_bpe32k.model`

Current token count:
- `3,556,233,011`

## Run Folders

Each experiment uses one stable folder under `runs\`.

Examples:
- `runs\rachana_gpt_tiny`
- `runs\rachana_llama_small`
- `runs\rachana_hybrid_mistral_tiny`

Each run folder stores:
- `train_log.csv`
- `cfg_snapshot.json`
- `run_metadata.json`
- `samples.txt`
- `checkpoints\latest`
- `checkpoints\best`
- `generation_eval\` after qualitative evaluation is run

## How Resume Works

All training scripts auto-resume from:

```text
runs\<run_name>\checkpoints\latest
```

To continue a run:
- keep the same `--run_dir`
- increase `--max_steps`

Example:
- first run to `5000`
- later continue to `10000`
- later continue to `15000`

## Post-Training Qualitative Evaluation

After a run reaches a meaningful milestone such as:
- `5000`
- `10000`
- `15000`

run the standardized generation evaluation harness before making comparison decisions.

Main files:
- [scripts/06_generate_eval_samples.py](C:/Users/prash/Projects/Rachana_GPT/scripts/06_generate_eval_samples.py)
- [eval_prompts/generation_prompts.jsonl](C:/Users/prash/Projects/Rachana_GPT/eval_prompts/generation_prompts.jsonl)
- [GENERATION_EVAL.md](C:/Users/prash/Projects/Rachana_GPT/GENERATION_EVAL.md)

This gives:
- same prompts across runs
- same temperatures across runs
- `best` and `latest` checkpoint comparison
- readable markdown plus structured JSON output

Recommended generation settings:
- temperatures: `0.7`, `0.9`, `1.1`
- `top_k = 50`
- `max_new_tokens = 200`

## Families

Current families:
- GPT
- LLaMA
- Mistral
- Hybrid LLaMA
- Hybrid Mistral

Hybrid notes:
- the hybrid trainer is [scripts/04_train_hf_hybrid.py](C:/Users/prash/Projects/Rachana_GPT/scripts/04_train_hf_hybrid.py)
- it now supports:
  - `--backbone llama`
  - `--backbone mistral`

## Current Preset Sizes

### Tiny
- GPT: `8L / 8H / 512`
- LLaMA: `8L / 8H / 512 / kv4 / ff1536`
- Mistral: same plus `sliding_window=512`
- Hybrid LLaMA: same as LLaMA-style hybrid
- Hybrid Mistral: same as Mistral-style hybrid

### Small
- GPT: `12L / 12H / 768`
- LLaMA: `12L / 12H / 768 / kv4 / ff2304`
- Mistral: same plus `sliding_window=768`
- Hybrid variants match the same parameter family

### Base
- GPT: `24L / 16H / 1024`
- LLaMA: `24L / 16H / 1024 / kv8 / ff3072`
- Mistral: same plus `sliding_window=1024`
- Hybrid variants match the same parameter family

## Suggested Training Order

Recommended order:

1. continue all `tiny` runs to the next target
2. run standardized generation evaluation on the same prompt set
3. compare them in the dashboards and markdown outputs
4. select the best families
5. continue or start `small`
6. only then move to `base`

## Commands

Use [commands.md](C:/Users/prash/Projects/Rachana_GPT/commands.md) as the main command reference.
