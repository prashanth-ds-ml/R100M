# Techniques Reference

This file summarizes the architecture techniques currently used in the training scripts and the most relevant additional architectures or upgrades worth testing next.

The goal is not to invent a brand-new model family. The goal is to combine proven techniques in a disciplined way so we can build the strongest Telugu-only base model possible under our hardware constraints.

## Current Families

| Family | Script | Core HF Model | Positional Method | Norm | MLP Style | Attention Style | KV Grouping | Sliding Window | Notes |
|---|---|---|---|---|---|---|---|---|---|
| GPT | `scripts/04_train_hf_gpt2.py` | `GPT2LMHeadModel` | Learned absolute positions | LayerNorm | GPT-style FFN | Full causal self-attention | No | No | Strong baseline, simple and stable |
| LLaMA | `scripts/04_train_hf_llama.py` | `LlamaForCausalLM` | RoPE | RMSNorm | SwiGLU-family via LLaMA implementation | Full causal self-attention | Yes via `n_kv_head` | No | Good modern baseline for efficient decoder training |
| Mistral | `scripts/04_train_hf_mistral.py` | `MistralForCausalLM` | RoPE | RMSNorm | SwiGLU-family via Mistral implementation | Full causal self-attention with Mistral stack | Yes via `n_kv_head` | Yes via `sliding_window` | Good efficiency/context candidate |
| Hybrid | `scripts/04_train_hf_hybrid.py` | `MistralForCausalLM` | RoPE | RMSNorm | SwiGLU-family | Causal self-attention | Yes, GQA-style by default | Yes | Pragmatic “best-techniques” line for this project |

## Training-Pipeline Techniques Shared Across Families

These are currently common across the training scripts:

| Technique | GPT | LLaMA | Mistral | Hybrid | Why it matters |
|---|---|---|---|---|---|
| Causal LM objective | Yes | Yes | Yes | Yes | Core next-token pretraining task |
| SentencePiece tokenizer | Yes | Yes | Yes | Yes | Shared tokenizer across architecture comparisons |
| One EOS per document | Yes | Yes | Yes | Yes | Preserves document boundaries cleanly |
| Packed token stream training | Yes | Yes | Yes | Yes | Efficient use of data and context window |
| Memmap token loading | Yes | Yes | Yes | Yes | Enables large-corpus training without loading everything into RAM |
| Auto-resume checkpoints | Yes | Yes | Yes | Yes | Makes long laptop-GPU runs practical |
| Train/val CSV logging | Yes | Yes | Yes | Yes | Keeps dashboard and comparison workflow consistent |
| Sample generation during training | Yes | Yes | Yes | Yes | Gives qualitative checkpoints |
| Gradient checkpointing option | Yes | Yes | Yes | Yes | Helps fit larger models on constrained VRAM |
| Mixed precision (`autocast`) | Yes | Yes | Yes | Yes | Improves throughput on CUDA |

## What Each Family Gives Us

### GPT

Best for:
- baseline comparisons
- simplest debugging path
- stable reference architecture

Weaknesses:
- older positional setup
- less modern efficiency than LLaMA/Mistral-family models

### LLaMA

Best for:
- strong modern baseline
- efficient parameter usage
- good Telugu-focused family comparison against GPT

Strengths:
- RoPE
- RMSNorm
- stronger MLP design than classic GPT blocks
- grouped-query support

### Mistral

Best for:
- efficiency-oriented experiments
- longer-context behavior
- testing whether sliding-window attention helps on document-style Telugu data

Strengths:
- all the main LLaMA-family upgrades
- sliding-window support
- grouped KV heads

### Hybrid

Best for:
- “best practical ingredients” experiment
- testing a strong small-model recipe without pretending to invent a new research architecture

Current hybrid ingredients:
- RoPE
- RMSNorm
- SwiGLU-style MLP
- grouped-query attention by default
- sliding-window support
- optional tied embeddings

## Additional Architecture Ideas Worth Testing

These are realistic next experiments.

| Idea | Worth testing? | Why |
|---|---|---|
| GQA vs full MHA | Yes | Important efficiency/quality tradeoff, especially on laptop GPU |
| Sliding window on/off | Yes | Useful to test whether document-style Telugu benefits from it at our context lengths |
| Embedding tying on/off | Yes | Can reduce params and sometimes help smaller models |
| Different MLP expansion ratios | Yes | Can improve parameter allocation at small scales |
| Different RoPE theta values | Maybe | Worth testing only after basic model-family comparisons are stable |
| Attention dropout > 0 | Maybe | Worth exploring if overfitting or instability appears |
| Larger context length with same params | Yes, later | Useful but expensive; better after corpus quality improves |

## Architectures We Could Explore Later

These are possible, but not top priority right now.

| Architecture / Idea | Recommendation | Reason |
|---|---|---|
| Qwen-style decoder stack | Later | Strong family, but adds another branch before we finish core comparisons |
| Phi-style small model design | Later | Interesting for compact models, but less urgent than GPT/LLaMA/Mistral/hybrid |
| RWKV / linear-attention families | Much later | Different training behavior and tooling; too much divergence right now |
| State-space models (Mamba etc.) | Much later | Interesting research path, but not the fastest route to a strong Telugu base model |
| MoE | No for now | Too complex and compute-fragmented for current setup |
| Encoder-decoder T5-style LM | No for base pretraining | Our current objective and downstream goals are decoder-first |

## Other Modern Families: Qwen, Phi, Gemma, Gemini

| Family | Priority | Why |
|---|---|---|
| Gemma | Later, but high-interest | Strong open family to compare after we finish the current 4-family baseline set |
| Qwen | Later | Strong modern family, but adds more tokenizer/recipe variance before we have stable baselines |
| Phi | Later | Very relevant for compact models, but strongly tied to data curation/training recipe quality |
| Gemini | Not in scope for direct family reproduction | Gemini is not an open training family for this repo in the same way as GPT/LLaMA/Mistral/Gemma |

Why `later` for Gemma/Qwen/Phi:

- We already have four meaningful families to compare.
- Adding too many families too early makes comparisons noisy.
- First we need clean answers on:
  - corpus quality
  - tokenizer quality
  - GPT vs LLaMA vs Mistral vs Hybrid behavior
- After that, adding one more family at a time becomes much more informative.

So the current project sequence should be:

1. finish the core four-family comparison
2. strengthen the hybrid through controlled ablations
3. then add `Gemma` first if we want a fifth family
4. then consider `Qwen`
5. then consider `Phi`

## Why Modern Techniques Matter

The goal is not “modern because fashionable.”
The goal is:

- better parameter efficiency
- more stable training
- better long-context behavior
- better small-model quality
- better hardware efficiency

For this project, modern techniques matter because we are trying to train the strongest possible Telugu-only model under constrained compute.

### Why not rely only on older GPT-style techniques?

Older GPT-style stacks are still useful as baselines, but they are weaker as the final target because they typically use:

- learned absolute positional embeddings
- LayerNorm instead of RMSNorm
- older FFN design instead of SwiGLU-family MLPs
- full attention everywhere without grouped KV efficiency
- less efficient context scaling behavior

These are not “bad,” but modern decoder stacks often give better quality-per-parameter and better practical scaling.

### Why use modern techniques?

| Technique | Why we want it |
|---|---|
| RoPE | Better positional handling and stronger extrapolation behavior than older learned absolute positions |
| RMSNorm | Simpler and often more stable/efficient than classic LayerNorm in modern decoder stacks |
| SwiGLU / modern MLP | Better parameter efficiency and stronger modeling capacity than older FFN choices |
| Grouped-query attention (GQA) | Reduces KV cost and improves efficiency, especially helpful on constrained hardware |
| Sliding-window attention | Can improve efficiency for longer contexts and document-style training |
| Gradient checkpointing | Makes larger modern stacks practical on limited VRAM |
| Packed document stream with EOS boundaries | Preserves document structure while staying compute-efficient |

## What We Need To Monitor When Using Modern Techniques

Modern techniques are useful only if they improve the run in practice. For every architecture line, we should monitor:

### Optimization and stability

- train loss
- val loss
- val perplexity
- grad norm
- OOM skips
- non-finite loss events

### Efficiency

- tokens per second
- seconds per update
- parameter count
- tokens per parameter
- memory pressure / whether the model fits comfortably

### Qualitative behavior

- repetition
- punctuation collapse
- topic drift
- long-form continuity
- Telugu fluency
- document-style coherence

### Comparative research signals

- same corpus, different architecture
- same parameter budget, different architecture
- same architecture, better corpus
- same corpus, different context length

These comparisons matter more than raw standalone loss numbers.

## Ideal Modern Decoder Configuration For This Project

This is the practical target configuration philosophy for a strong Telugu base model under our current constraints.

### Ideal ingredients

| Component | Ideal choice | Why |
|---|---|---|
| Model family | Decoder-only causal LM | Matches current base-model and RAG goals |
| Positions | RoPE | Better modern default than learned absolute positions |
| Normalization | RMSNorm | Modern stable decoder default |
| MLP | SwiGLU-family | Better parameter efficiency |
| Attention | Causal self-attention | Standard base-model training objective |
| KV heads | GQA-style (`num_key_value_heads < num_attention_heads`) | Better efficiency on smaller hardware |
| Context efficiency | Sliding-window support | Good option for document-style training and future longer contexts |
| EOS policy | One EOS per document | Best for document-level training on Sangraha-style data |
| Tokenizer | SentencePiece Telugu-trained tokenizer | Good fit for Telugu-only corpus and current pipeline |
| Checkpointing | Gradient checkpointing enabled when needed | Helps fit bigger models on laptop GPU |

### Ideal hybrid target for this repo

The ideal hybrid target is:

- decoder-only
- RoPE
- RMSNorm
- SwiGLU-family MLP
- grouped-query attention by default
- sliding-window support
- packed token-stream training with document EOS boundaries
- clean Telugu tokenizer and corpus

This is exactly the kind of “best known practical techniques” stack we want.

## Are We Using The Ideal Hybrid Ingredients Right Now?

### Current `scripts/04_train_hf_hybrid.py`

| Technique | Using now? | Notes |
|---|---|---|
| Decoder-only causal LM | Yes | Uses `MistralForCausalLM` |
| RoPE | Yes | Via Mistral-family config |
| RMSNorm | Yes | Via Mistral-family stack |
| SwiGLU-family MLP | Yes | Via Mistral-family implementation |
| Grouped-query attention | Yes, by default | Default `n_kv_head = max(1, n_head // 4)` unless overridden |
| Sliding-window support | Yes | `sliding_window` is configurable |
| SentencePiece tokenizer | Yes | Same tokenizer pipeline as other models |
| One EOS per document | Yes | Shared tokenization pipeline behavior |
| Packed token stream | Yes | Shared memmap training flow |
| Gradient checkpointing | Yes | Optional via `--grad_ckpt` |
| Attention dropout control | Yes | Configurable |
| Embedding tying | Optional | Available, but off by default |

### So are we already close to the ideal?

Yes, architecturally we are already very close to the practical target.

What still matters more than adding extra architecture tricks:

- better corpus quality
- cleaner `base_corpus_v2`
- fair matched experiments
- good eval discipline
- small ablations on the hybrid line

## Recommended Final Hybrid Configuration To Aim For

This is the configuration philosophy, not one magic number set for every GPU:

### Structural config

- family: hybrid decoder-only
- positions: RoPE
- norm: RMSNorm
- MLP: SwiGLU-family
- attention: causal
- KV heads: grouped-query attention
- sliding window: enabled
- tokenizer: Telugu SentencePiece
- EOS: one per cleaned document

### Practical training defaults

- gradient checkpointing: on when VRAM is tight
- attention dropout: `0.0` or very low unless needed for stability
- tie word embeddings: experiment-dependent, not mandatory
- context length: start with `512`, move to `768` only when the model/corpus setup is stable

### Recommended default hybrid shape for current experiments

For constrained hardware:

- `n_layer`: 8 to 12 for tiny/small pilots
- `n_head`: 8 to 12
- `n_embd`: 512 to 768
- `n_kv_head`: around `n_head // 4` or `n_head // 2`
- `intermediate_size`: LLaMA/Mistral-style derived default

For a stronger later base run:

- same ingredient stack
- larger depth/width only after:
  - `base_corpus_v2` is ready
  - eval pool is defined
  - family comparisons are complete

## What To Improve Next In The Hybrid Line

The next useful improvements are not new families. They are controlled hybrid ablations:

1. GQA ratio:
   - compare `n_kv_head = n_head`
   - vs `n_kv_head = n_head // 2`
   - vs `n_kv_head = n_head // 4`

2. Sliding window:
   - on vs off

3. Embedding tying:
   - on vs off

4. MLP expansion:
   - current derived default vs a slightly larger ratio

5. Context length:
   - `512` vs `768` after corpus quality is improved

That is the best path to making the hybrid stronger without losing research clarity.

## Recommended Research Sequence

The strongest sequence for this repo is:

1. Keep GPT as the reference baseline.
2. Run LLaMA as the first modern comparison.
3. Run Mistral as the efficiency/context comparison.
4. Run Hybrid as the “best practical recipe” line.
5. Compare all four under matched token/corpus budgets.
6. Only then start adding smaller upgrades such as:
   - GQA ablations
   - sliding-window ablations
   - embedding tying
   - MLP expansion changes

## Recommendation For The Current Project

The project should stay focused on:

- strongest Telugu-only corpus
- reproducible training pipeline
- matched architecture comparisons
- one pragmatic hybrid line with the best currently known small-model techniques

That is enough to make the project research-grade without turning it into unfocused architecture sprawl.
