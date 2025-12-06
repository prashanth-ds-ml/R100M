# R100M – RachanaLM: A Small, Strong Telugu Language Model

R100M (RachanaLM-100M) is a research project to build a **small but high-quality Telugu language model** that:

- Trains primarily on a **single consumer GPU** (RTX 3060 6 GB)
- Handles **pure Telugu, roman Telugu, and code-mixed Telugu–English**
- Learns **entity awareness** using NER supervision
- Gains better **reasoning and comprehension** via synthetic tasks
- Can be used both as a **generator** and as a **RAG embedding encoder**

The long-term goal is to show that **careful data curation + auxiliary tasks** can make a ~100M parameter Telugu model **competitive with much larger generic LLMs** on Telugu-specific tasks, at a fraction of the compute cost.


## Repository Layout

- `data/` – raw and processed corpora, tokenizer corpus, binary token files
- `r100m/` – Python package with configs, model, training, eval and inference code
- `scripts/` – CLI scripts to prepare data and run training
- `notebooks/` – exploratory notebooks for cleaning, tokenizer tests, sanity checks
- `docs/` – project plan, research log, paper outline
- `tests/` – basic unit tests for tokenizer, model shapes, etc.

## Project Status

- [\u2705] Phase 0 – Project skeleton & repo structure
- [ ] Phase 1 – Data cleaning pipeline for Sangraha (Telugu)
- [ ] Phase 2 – Tokenizer training & HF release
- [ ] Phase 3 – RachanaLM-100M base pretraining (v0.1)
- [ ] Phase 4 – Roman Telugu & code-mix robustness (v0.2)
- [ ] Phase 5 – NER multi-task & entity awareness (v0.3)
- [ ] Phase 6 – Reasoning & comprehension (v0.4)
- [ ] Phase 7 – RAG embedding model (v1.0)
- [ ] Phase 8 – A100 scaling experiments & paper

This repo is in **active development**. All experiments and decisions will be logged in `docs/research_log.md`.
