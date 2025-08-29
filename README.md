# oss-benchmark
Benchmarking OSS model performance on-device for medical tasks.

### Setup
```
pip install cloudscraper beautifulsoup4
```

### Benchmarking Eurorad
Official OpenAI API example, GPT-4o batch
```
python benchmarks/eurorad/gpt.py --mode batch --debug
```

GPT-OSS example (official HuggingFace API)
```
python benchmarks/eurorad/hf_bench.py \
  /home/bowang/Documents/alif/oss-benchmark/data/datasets/eurorad_test.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort medium \
  --max_output_tokens 8192 \
  --workers 1 \
  --results /home/bowang/Documents/alif/oss-benchmark/results
```

Openrouter Example
```
python benchmarks/eurorad/openrouter.py \
  /home/bowang/Documents/alif/oss-benchmark/data/datasets/eurorad_test.csv \
  --endpoint qwen/qwen3-30b-a3b-instruct-2507 \
  --results_dir /home/bowang/Documents/alif/oss-benchmark/results \
  --max_output_tokens 8192 \
  --workers 1 \
  --resume
```

### Benchmarking Ophthalmology

gpt-4o / gpt-5
```
# Chat Completions over all questions (gpt-5)
python benchmarks/ophthalmology/gpt.py --mode chat --chat-model gpt-5 --debug

# Chat Completions over all questions (gpt-4o)
python benchmarks/ophthalmology/gpt.py --mode chat --chat-model gpt-4o --debug

# Responses API (gpt-5) with reasoning control
python benchmarks/ophthalmology/gpt.py --mode responses --effort low --debug

# Batch API (gpt-4o)
python benchmarks/ophthalmology/gpt.py --mode batch --debug
```

gpt-oss-20b
```
# First run
python benchmarks/ophthalmology/hf_bench.py data/datasets/ophthalmology.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort low \
  --max_output_tokens 8192 \
  --workers 2 \
  --results results

# Resume
python benchmarks/ophthalmology/hf_bench.py results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-low_max8192.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort low \
  --max_output_tokens 8192 \
  --workers 2 \
  --results results \
  --resume \
  --output_csv results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-low_max8192.csv
```

gpt-oss-120b
```
# First run
python benchmarks/ophthalmology/hf_bench.py data/datasets/ophthalmology.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat \
  --reasoning_effort high \
  --max_output_tokens 8192 \
  --workers 2 \
  --results results

# Resume
python benchmarks/ophthalmology/hf_bench.py results/ophthalmology_openai-gpt-oss-120b-fireworks-ai_chat_re-high_max8192.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat \
  --reasoning_effort high \
  --max_output_tokens 8192 \
  --workers 2 \
  --results results \
  --resume \
  --output_csv results/ophthalmology_openai-gpt-oss-120b-fireworks-ai_chat_re-high_max8192.csv
```


HuggingFace
```
python benchmarks/ophthalmology/hf_bench_dataset.py results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-high_max2048.csv \
  --model openai/gpt-oss-20b:fireworks-ai \
  --api chat \
  --reasoning_effort high \
  --max_output_tokens 4096 \
  --workers 1 \
  --results results \
  --resume \
  --output_csv results/ophthalmology_openai-gpt-oss-20b-fireworks-ai_chat_re-high_max2048.csv \
  --max_retries 8 --base_backoff 5
```

```
python benchmarks/ophthalmology/hf_bench.py results/ophthalmology_openai-gpt-oss-120b-fireworks-ai_chat_re-low_max4096.csv \
  --model openai/gpt-oss-120b:fireworks-ai \
  --api chat \
  --reasoning_effort low \
  --max_output_tokens 4096 \
  --workers 2 \
  --results results \
  --resume \
  --output_csv results/ophthalmology_openai-gpt-oss-120b-fireworks-ai_chat_re-low_max4096.csv
```

Novita
```
python benchmarks/ophthalmology/novita.py data/datasets/ophthalmology.csv \
  --endpoint baichuan/baichuan-m2-32b \
  --sleep 1.2 --timeout 60 --verbose --resume
```

OpenRouter
```
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint z-ai/glm-4.5-air \
  --results_dir results \
  --workers 1 \
  --resume
```

OpenAI
```
python benchmarks/ophthalmology/openai_bench.py data/datasets/ophthalmology.csv \
  --model gpt-oss-20b \
  --results_dir results \
  --reasoning_effort high \
  --max_output_tokens 2048
```

```
# Basic: endpoint, auto-named CSV goes into results/
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint qwen/qwen3-30b-a3b-instruct-2507

# Explicit results directory and more workers
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint qwen/qwen3-30b-a3b-instruct-2507 \
  --results_dir results --workers 8 --temperature 0.0

# Resume a previous run and cap to first 100 rows
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint qwen/qwen3-30b-a3b-instruct-2507 \
  --resume --max 100

# Override prompts if you need non-MCQ behavior later
python benchmarks/ophthalmology/openrouter.py data/datasets/ophthalmology.csv \
  --endpoint meta-llama/llama-3.1-8b-instruct \
  --system "Answer succinctly with one token." \
  --user_template "{case_text}\nReturn one word only."
```

### Data

python data/csvs_to_json.py --indir eurorad_csvs --out eurorad_cases.json

python data/combine_cases_csv.py --indir eurorad_csvs --out eurorad_cases_wide.csv


Get a single case
```
python data/get_case_eurorad.py https://www.eurorad.org/case/18706
```

Get all 2025 cases
```
python data/get_range_eurorad.py --start 18806 --end 19164 --outdir eurorad_csvs
```