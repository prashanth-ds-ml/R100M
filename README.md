# Rachana GPT

Rachana GPT is a Telugu-first base language model project focused on:
- building a high-quality Telugu pretraining corpus
- tokenizing it with SentencePiece
- training and comparing multiple decoder-only model families
- monitoring runs locally through dashboards and logs

## Current Project State

Current tokenized pretraining corpus:
- file: [data/rachana_tokens/tokens.bin](C:/Users/prash/Projects/Rachana_GPT/data/rachana_tokens/tokens.bin)
- token count: `3,556,233,011`
- tokenizer: [tokenizer/rachana_bpe32k.model](C:/Users/prash/Projects/Rachana_GPT/tokenizer/rachana_bpe32k.model)

Current final text corpus:
- file: [data/final/final_pretrain_corpus_v2_sangraha76.txt](C:/Users/prash/Projects/Rachana_GPT/data/final/final_pretrain_corpus_v2_sangraha76.txt)
- metadata: [data/final/final_pretrain_corpus_v2_sangraha76.meta.json](C:/Users/prash/Projects/Rachana_GPT/data/final/final_pretrain_corpus_v2_sangraha76.meta.json)

Families currently supported:
- GPT
- LLaMA
- Mistral
- Hybrid with LLaMA backbone
- Hybrid with Mistral backbone

## Main Scripts

Data and token pipeline:
- [scripts/01_tokenize_eos.py](C:/Users/prash/Projects/Rachana_GPT/scripts/01_tokenize_eos.py)
- [scripts/02_make_memmap.py](C:/Users/prash/Projects/Rachana_GPT/scripts/02_make_memmap.py)
- [scripts/03_decode_token_samples.py](C:/Users/prash/Projects/Rachana_GPT/scripts/03_decode_token_samples.py)

Training:
- [scripts/04_train_hf_gpt2.py](C:/Users/prash/Projects/Rachana_GPT/scripts/04_train_hf_gpt2.py)
- [scripts/04_train_hf_llama.py](C:/Users/prash/Projects/Rachana_GPT/scripts/04_train_hf_llama.py)
- [scripts/04_train_hf_mistral.py](C:/Users/prash/Projects/Rachana_GPT/scripts/04_train_hf_mistral.py)
- [scripts/04_train_hf_hybrid.py](C:/Users/prash/Projects/Rachana_GPT/scripts/04_train_hf_hybrid.py)
- [scripts/05_run_experiment_queue.py](C:/Users/prash/Projects/Rachana_GPT/scripts/05_run_experiment_queue.py)

Dashboards:
- [dashboard_live/server.py](C:/Users/prash/Projects/Rachana_GPT/dashboard_live/server.py)
- [dashboard_research/server.py](C:/Users/prash/Projects/Rachana_GPT/dashboard_research/server.py)
- [dashboard_html/build_dashboard.py](C:/Users/prash/Projects/Rachana_GPT/dashboard_html/build_dashboard.py)

## Documentation

Main runbook:
- [commands.md](C:/Users/prash/Projects/Rachana_GPT/commands.md)

Training guide:
- [TRAINING.md](C:/Users/prash/Projects/Rachana_GPT/TRAINING.md)

Dashboard guide:
- [DASHBOARDS.md](C:/Users/prash/Projects/Rachana_GPT/DASHBOARDS.md)

Experiment presets:
- [EXPERIMENTS.md](C:/Users/prash/Projects/Rachana_GPT/EXPERIMENTS.md)

Generation evaluation:
- [GENERATION_EVAL.md](C:/Users/prash/Projects/Rachana_GPT/GENERATION_EVAL.md)

Architecture notes:
- [TECHNIQUES.md](C:/Users/prash/Projects/Rachana_GPT/TECHNIQUES.md)

## Recommended Workflow

1. Activate the `llm` env.
2. Start the live dashboard.
3. Train one model at a time with the commands in [commands.md](C:/Users/prash/Projects/Rachana_GPT/commands.md).
4. Resume the same run by keeping the same `run_dir` and increasing `max_steps`.
5. Compare runs in the research dashboard.

## Note

Some older markdown files in the repo are historical and may not reflect the latest current training setup as accurately as:
- [README.md](C:/Users/prash/Projects/Rachana_GPT/README.md)
- [TRAINING.md](C:/Users/prash/Projects/Rachana_GPT/TRAINING.md)
- [DASHBOARDS.md](C:/Users/prash/Projects/Rachana_GPT/DASHBOARDS.md)
- [EXPERIMENTS.md](C:/Users/prash/Projects/Rachana_GPT/EXPERIMENTS.md)
- [commands.md](C:/Users/prash/Projects/Rachana_GPT/commands.md)
