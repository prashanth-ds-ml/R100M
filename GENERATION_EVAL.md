# Generation Evaluation

This project now supports a standardized qualitative generation workflow.

Purpose:
- compare the same prompts across models
- compare `best` and `latest` checkpoints
- compare multiple temperatures for the same prompt

## Main Script

- [scripts/06_generate_eval_samples.py](C:/Users/prash/Projects/Rachana_GPT/scripts/06_generate_eval_samples.py)

## Prompt File

- [eval_prompts/generation_prompts.jsonl](C:/Users/prash/Projects/Rachana_GPT/eval_prompts/generation_prompts.jsonl)

Prompt categories currently included:
- story
- news
- article
- essay
- dialogue
- culture

## Default Evaluation Settings

Default settings in the script:
- checkpoints: `best` and `latest`
- temperatures: `0.7,0.9,1.1`
- `top_k = 50`
- `max_new_tokens = 200`

These settings are intended to test:
- lower temperature coherence
- medium temperature naturalness
- higher temperature creativity and stability

## Example Command

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

## Output

Per run, outputs are written to:

```text
runs\<run_name>\generation_eval\
```

Files:
- `best_generations.json`
- `best_generations.md`
- `latest_generations.json`
- `latest_generations.md`

## Recommended Use

Run this after a meaningful training point such as:
- `5000`
- `10000`
- `15000`

Then compare:
- coherence
- Telugu fluency
- repetition
- prompt adherence
- genre/style continuation

This should be used alongside the training dashboards, not as a replacement for them.
