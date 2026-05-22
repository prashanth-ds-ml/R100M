# Rachana GPT Commands

This is the working command reference for the current codebase.

Use this file when you want to:
- start dashboards
- inspect token artifacts
- train one model at a time
- continue an existing run by increasing `--max_steps`

Current corpus / token state:
- final corpus: `data\final\final_pretrain_corpus_v2_sangraha76.txt`
- token file: `data\rachana_tokens\tokens.bin`
- token metadata: `data\rachana_tokens\meta.json`
- current token count: `3,556,233,011`

## Environment

```powershell
conda activate llm
cd C:\Users\prash\Projects\Rachana_GPT
```

## Dashboards

### Live dashboard

```powershell
python .\dashboard_live\server.py
```

Open:

```text
http://127.0.0.1:8765
```

### Research dashboard

```powershell
python .\dashboard_research\server.py
```

Open:

```text
http://127.0.0.1:8770
```

### Static HTML dashboard

```powershell
python .\dashboard_html\build_dashboard.py
```

## Token / Data Checks

### Check memmap

```powershell
python .\scripts\02_make_memmap.py
```

### Decode sample documents

```powershell
python .\scripts\03_decode_token_samples.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --first_docs 5 `
  --random_docs 5 `
  --max_decode_tokens 512 `
  --excerpt_chars 1200
```

### Standardized generation evaluation from a trained run

```powershell
python .\scripts\06_generate_eval_samples.py `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_llama_tiny" `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --prompt_file "C:\Users\prash\Projects\Rachana_GPT\eval_prompts\generation_prompts.jsonl" `
  --checkpoint both `
  --temperatures 0.7,0.9,1.1 `
  --top_k 50 `
  --max_new_tokens 200
```

## Important Training Note

All trainer scripts auto-resume from:

```text
runs\<run_name>\checkpoints\latest
```

So to continue training the same model:
- keep the same `--run_dir`
- increase `--max_steps`

Example:
- first run to `5000`
- later continue to `10000`
- later continue to `15000`

## Individual Training Commands

These commands are for running models separately instead of using the automation queue.

---

## Tiny Runs

### GPT tiny

```powershell
python .\scripts\04_train_hf_gpt2.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_gpt_tiny" `
  --max_length 512 `
  --stride 512 `
  --n_layer 8 `
  --n_head 8 `
  --n_embd 512 `
  --batch_size 1 `
  --grad_accum 16 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 100 `
  --eval_every_updates 100 `
  --val_fraction 0.005
```

### LLaMA tiny

```powershell
python .\scripts\04_train_hf_llama.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_llama_tiny" `
  --max_length 512 `
  --stride 512 `
  --n_layer 8 `
  --n_head 8 `
  --n_embd 512 `
  --n_kv_head 4 `
  --intermediate_size 1536 `
  --batch_size 1 `
  --grad_accum 16 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 100 `
  --eval_every_updates 100 `
  --val_fraction 0.005
```

### Mistral tiny

```powershell
python .\scripts\04_train_hf_mistral.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_mistral_tiny" `
  --max_length 512 `
  --stride 512 `
  --n_layer 8 `
  --n_head 8 `
  --n_embd 512 `
  --n_kv_head 4 `
  --intermediate_size 1536 `
  --sliding_window 512 `
  --batch_size 1 `
  --grad_accum 16 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 100 `
  --eval_every_updates 100 `
  --val_fraction 0.005
```

### Hybrid LLaMA tiny

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_llama_tiny" `
  --backbone llama `
  --max_length 512 `
  --stride 512 `
  --n_layer 8 `
  --n_head 8 `
  --n_embd 512 `
  --n_kv_head 4 `
  --intermediate_size 1536 `
  --batch_size 1 `
  --grad_accum 16 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 100 `
  --eval_every_updates 100 `
  --val_fraction 0.005
```

### Hybrid Mistral tiny

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_mistral_tiny" `
  --backbone mistral `
  --max_length 512 `
  --stride 512 `
  --n_layer 8 `
  --n_head 8 `
  --n_embd 512 `
  --n_kv_head 4 `
  --intermediate_size 1536 `
  --sliding_window 512 `
  --batch_size 1 `
  --grad_accum 16 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 100 `
  --eval_every_updates 100 `
  --val_fraction 0.005
```

---

## Small Runs

### GPT small

```powershell
python .\scripts\04_train_hf_gpt2.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_gpt_small" `
  --max_length 768 `
  --stride 768 `
  --n_layer 12 `
  --n_head 12 `
  --n_embd 768 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 200 `
  --eval_every_updates 200 `
  --val_fraction 0.005
```

### LLaMA small

```powershell
python .\scripts\04_train_hf_llama.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_llama_small" `
  --max_length 768 `
  --stride 768 `
  --n_layer 12 `
  --n_head 12 `
  --n_embd 768 `
  --n_kv_head 4 `
  --intermediate_size 2304 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 200 `
  --eval_every_updates 200 `
  --val_fraction 0.005
```

### Mistral small

```powershell
python .\scripts\04_train_hf_mistral.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_mistral_small" `
  --max_length 768 `
  --stride 768 `
  --n_layer 12 `
  --n_head 12 `
  --n_embd 768 `
  --n_kv_head 4 `
  --intermediate_size 2304 `
  --sliding_window 768 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 200 `
  --eval_every_updates 200 `
  --val_fraction 0.005
```

### Hybrid LLaMA small

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_llama_small" `
  --backbone llama `
  --max_length 768 `
  --stride 768 `
  --n_layer 12 `
  --n_head 12 `
  --n_embd 768 `
  --n_kv_head 4 `
  --intermediate_size 2304 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 200 `
  --eval_every_updates 200 `
  --val_fraction 0.005
```

### Hybrid Mistral small

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_mistral_small" `
  --backbone mistral `
  --max_length 768 `
  --stride 768 `
  --n_layer 12 `
  --n_head 12 `
  --n_embd 768 `
  --n_kv_head 4 `
  --intermediate_size 2304 `
  --sliding_window 768 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 200 `
  --eval_every_updates 200 `
  --val_fraction 0.005
```

---

## Base Runs

### GPT base

```powershell
python .\scripts\04_train_hf_gpt2.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_gpt_base" `
  --max_length 1024 `
  --stride 1024 `
  --n_layer 24 `
  --n_head 16 `
  --n_embd 1024 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 250 `
  --eval_every_updates 250 `
  --val_fraction 0.005
```

### LLaMA base

```powershell
python .\scripts\04_train_hf_llama.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_llama_base" `
  --max_length 1024 `
  --stride 1024 `
  --n_layer 24 `
  --n_head 16 `
  --n_embd 1024 `
  --n_kv_head 8 `
  --intermediate_size 3072 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 250 `
  --eval_every_updates 250 `
  --val_fraction 0.005
```

### Mistral base

```powershell
python .\scripts\04_train_hf_mistral.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_mistral_base" `
  --max_length 1024 `
  --stride 1024 `
  --n_layer 24 `
  --n_head 16 `
  --n_embd 1024 `
  --n_kv_head 8 `
  --intermediate_size 3072 `
  --sliding_window 1024 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 250 `
  --eval_every_updates 250 `
  --val_fraction 0.005
```

### Hybrid LLaMA base

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_llama_base" `
  --backbone llama `
  --max_length 1024 `
  --stride 1024 `
  --n_layer 24 `
  --n_head 16 `
  --n_embd 1024 `
  --n_kv_head 8 `
  --intermediate_size 3072 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 250 `
  --eval_every_updates 250 `
  --val_fraction 0.005
```

### Hybrid Mistral base

```powershell
python .\scripts\04_train_hf_hybrid.py `
  --tok_dir "C:\Users\prash\Projects\Rachana_GPT\data\rachana_tokens" `
  --sp_model_path "C:\Users\prash\Projects\Rachana_GPT\tokenizer\rachana_bpe32k.model" `
  --run_dir "C:\Users\prash\Projects\Rachana_GPT\runs\rachana_hybrid_mistral_base" `
  --backbone mistral `
  --max_length 1024 `
  --stride 1024 `
  --n_layer 24 `
  --n_head 16 `
  --n_embd 1024 `
  --n_kv_head 8 `
  --intermediate_size 3072 `
  --sliding_window 1024 `
  --batch_size 1 `
  --grad_accum 32 `
  --max_steps 15000 `
  --grad_ckpt `
  --save_every_updates 250 `
  --eval_every_updates 250 `
  --val_fraction 0.005
```

## Automation Queue

The queue script still exists if you want it:

```powershell
python .\scripts\05_run_experiment_queue.py `
  --families gpt,llama,mistral,hybrid_llama,hybrid_mistral `
  --scales tiny `
  --dry_run
```

But if training time is stretching and you want tighter control, prefer the individual commands above.
