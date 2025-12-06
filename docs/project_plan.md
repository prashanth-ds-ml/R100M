# R100M / RachanaLM – Project Plan

## 1. High-Level Goal

Build a **~100M parameter Telugu language model** (RachanaLM-100M) that:

1. Trains primarily on a **single RTX 3060 6 GB GPU** with minimal paid compute.
2. Achieves **fluent, natural Telugu** generation.
3. Robustly understands **roman Telugu** and **Telugu–English code-mix**.
4. Is **entity-aware**, using AI4Bharat NER data as auxiliary supervision.
5. Shows improved **reasoning and comprehension** using synthetic datasets.
6. Can serve as both a **generator** and a **RAG embedding encoder**.
7. Competes with larger generic models on Telugu-specific evaluation tasks.

## 2. Phases & Milestones

### Phase 1 – Data & Cleaning
- Identify and download datasets:
  - Sangraha (Telugu subset)
  - AI4Bharat NER (Telugu)
  - Sources for code-mix and roman Telugu
  - Synthetic reasoning / QA data (to be generated)
- Implement canonical cleaning pipeline:
  - Unicode normalization
  - Noise filtering
  - Leading punctuation / junk removal
  - Global deduplication
- Output:
  - `data/cleaned/sangraha_telugu.txt`
  - `data/metadata/cleaning_report.md`

### Phase 2 – Tokenizer
- Curate balanced tokenizer corpus:
  - Telugu script
  - Code-mixed Tel–Eng
  - Roman Telugu variants
- Train SentencePiece tokenizer (24k–32k vocab).
- Evaluate tokenizer:
  - Tokenization examples
  - Length statistics for different domains
- Publish tokenizer to Hugging Face.
- Output:
  - `tokenizer/sp.model`, `sp.vocab`
  - HF repo `Vipplav/RachanaLM-tokenizer` (name TBD)

### Phase 3 – Base LM (v0.1)
- Define RachanaLM-60M and RachanaLM-100M configs.
- Implement model in `r100m/models/rachana_lm.py`.
- Convert cleaned corpus to binary token format (`.bin`).
- Train base LM (pure LM loss) on RTX 3060:
  - Target token budget: 100M–300M tokens.
- Evaluate:
  - Perplexity on held-out Telugu.
  - Qualitative generation samples.

### Phase 4 – Roman Telugu & Code-Mix (v0.2)
- Generate roman Telugu text via transliteration.
- Introduce code-mix and roman text into training (corpus or on-the-fly).
- Continue training from v0.1.
- Evaluate:
  - Perplexity on roman Telugu test set.
  - Qualitative behaviour on code-mixed prompts.

### Phase 5 – NER Multi-Task & Entity Awareness (v0.3)
- Prepare AI4Bharat NER dataset with subword label alignment.
- Implement NER head and multi-task training (LM + NER loss).
- Train from v0.2 checkpoint.
- Evaluate:
  - NER F1 on held-out data.
  - Entity consistency in generations.

### Phase 6 – Reasoning & Comprehension (v0.4)
- Build synthetic reasoning datasets (using external LLMs).
- Add reasoning SFT / multi-task finetuning.
- Evaluate:
  - QA performance (JNANA subset, synthetic QA).
  - Reasoning examples and human-rated quality.

### Phase 7 – RAG Embedding Model (v1.0)
- Define embedding extraction (pooling over LM hidden states).
- Prepare contrastive retrieval dataset: (question, passage) pairs.
- Train RAG encoder (RachanaLM-Embed-100M).
- Evaluate retrieval (recall@k) vs standard multilingual embedding models.

### Phase 8 – A100 Scaling & Paper
- Optional: train RachanaLM-250M on A100.
- Compare models across:
  - Size, tokens, compute cost, performance.
- Write and publish research paper (Zenodo) and blog / LinkedIn summary.

## 3. Tracking

All major experiments, configs, and observations will be recorded in `docs/research_log.md` and in Weights & Biases runs tagged with:

- `model_size` (60M, 100M, 250M)
- `phase` (v0.1, v0.2, v0.3, v0.4, embed)
- `data_variant` (base, translit, ner, reasoning)
