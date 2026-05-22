# Experiment Presets

This file records the current experiment naming and preset sizes used by:
- [scripts/05_run_experiment_queue.py](C:/Users/prash/Projects/Rachana_GPT/scripts/05_run_experiment_queue.py)
- the individual commands in [commands.md](C:/Users/prash/Projects/Rachana_GPT/commands.md)

## Naming Convention

Current naming:
- `rachana_gpt_tiny`
- `rachana_llama_tiny`
- `rachana_mistral_tiny`
- `rachana_hybrid_llama_tiny`
- `rachana_hybrid_mistral_tiny`

Likewise for:
- `small`
- `base`

Examples:
- `rachana_gpt_small`
- `rachana_llama_small`
- `rachana_hybrid_mistral_base`

## Preset Sizes

## Tiny

Common:
- `max_length=512`
- `stride=512`
- `batch_size=1`
- `grad_accum=16`
- `save_every_updates=100`
- `eval_every_updates=100`

Per family:
- GPT: `n_layer=8`, `n_head=8`, `n_embd=512`
- LLaMA: `n_layer=8`, `n_head=8`, `n_embd=512`, `n_kv_head=4`, `intermediate_size=1536`
- Mistral: same as LLaMA plus `sliding_window=512`
- Hybrid LLaMA: same as LLaMA-style hybrid
- Hybrid Mistral: same as Mistral-style hybrid

## Small

Common:
- `max_length=768`
- `stride=768`
- `batch_size=1`
- `grad_accum=32`
- `save_every_updates=200`
- `eval_every_updates=200`

Per family:
- GPT: `n_layer=12`, `n_head=12`, `n_embd=768`
- LLaMA: `n_layer=12`, `n_head=12`, `n_embd=768`, `n_kv_head=4`, `intermediate_size=2304`
- Mistral: same as LLaMA plus `sliding_window=768`
- Hybrid LLaMA: same as LLaMA-style hybrid
- Hybrid Mistral: same as Mistral-style hybrid

## Base

Common:
- `max_length=1024`
- `stride=1024`
- `batch_size=1`
- `grad_accum=32`
- `save_every_updates=250`
- `eval_every_updates=250`

Per family:
- GPT: `n_layer=24`, `n_head=16`, `n_embd=1024`
- LLaMA: `n_layer=24`, `n_head=16`, `n_embd=1024`, `n_kv_head=8`, `intermediate_size=3072`
- Mistral: same as LLaMA plus `sliding_window=1024`
- Hybrid LLaMA: same as LLaMA-style hybrid
- Hybrid Mistral: same as Mistral-style hybrid

## Step Schedule

Current practice:
- train in increments of `5000` steps
- resume the same run instead of starting a new folder

Examples:
- `5000`
- `10000`
- `15000`
- `20000`

## Qualitative Evaluation Milestones

After each major step target, run qualitative generation evaluation.

Recommended checkpoints:
- after `5000`
- after `10000`
- after `15000`

Use:
- [scripts/06_generate_eval_samples.py](C:/Users/prash/Projects/Rachana_GPT/scripts/06_generate_eval_samples.py)
- [eval_prompts/generation_prompts.jsonl](C:/Users/prash/Projects/Rachana_GPT/eval_prompts/generation_prompts.jsonl)

Recommended qualitative settings:
- temperatures: `0.7`, `0.9`, `1.1`
- `top_k = 50`
- `max_new_tokens = 200`

Compare:
- `best` vs `latest`
- same prompt across families
- same temperature across families

Focus on:
- Telugu fluency
- coherence
- repetition
- prompt adherence
- style continuation

## Recommended Comparison Flow

1. compare all `tiny` runs
2. run standardized generation evaluation for each run
3. promote the best 2-3 families to `small`
4. repeat the same evaluation cycle
5. promote the strongest families to `base`
