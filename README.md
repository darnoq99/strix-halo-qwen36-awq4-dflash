# Qwen3.6-27B AWQ4 + DFlash on AMD Strix Halo

Reproducible notes and scripts for running `Qwen3.6-27B-AWQ4` with DFlash speculative decoding on AMD Strix Halo / Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) using vLLM, ROCm, AWQ4, and a full 256K context window.

This repository is a field report plus launcher, not a fork of vLLM. The container build comes from the Strix Halo-specific upstream project:

- <https://github.com/hec-ovi/vllm-awq4-qwen>
- <https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4>
- <https://huggingface.co/z-lab/Qwen3.6-27B-DFlash>
- <https://github.com/vllm-project/vllm/pull/40898>

## Result Summary

Best local variant:

- Model: `Qwen3.6-27B-AWQ4`
- Target weights: `cyankiwi/Qwen3.6-27B-AWQ-INT4`
- Drafter: `z-lab/Qwen3.6-27B-DFlash`
- Context: `262144`
- vLLM path: `/v1/responses`, `stream: true`
- Reasoning disabled for agent throughput: `chat_template_kwargs.enable_thinking=false`
- `MAX_NUM_SEQS=3`
- `DFLASH_TOKENS=8`
- `MAX_BATCHED_TOKENS=8192`
- `--safetensors-load-strategy eager`

Measured local result:

```text
3 concurrent long generations:
5924 output tokens / 134.946s = 43.90 tok/s aggregate
per stream: 14.3-15.3 tok/s
TTFT: about 0.015s
reasoning chars: 0
```

Single-request throughput did not reach 40-50 tok/s. On this machine, 40-50 tok/s was reached as aggregate throughput with three concurrent streams.

## Repository Contents

```text
scripts/run_best_awq4_dflash_server.sh
benchmarks/stream_bench.py
benchmarks/stream_tuning_results.jsonl
reports/SPEED_TUNING_REPORT_2026-04-29.md
.env.example
```

## Reproduce The Environment

These steps assume Fedora or another Linux host with working Strix Halo ROCm device access through `/dev/kfd` and `/dev/dri`, plus Podman.

Use a large disk. The examples use `/mnt/bigdrive`.

### 1. Create work directories

```bash
mkdir -p /mnt/bigdrive/strix-qwen36-vllm-awq4-qwen
mkdir -p /mnt/bigdrive/strix-qwen36-vllm-dflash/models
```

### 2. Clone and build the Strix Halo vLLM container

```bash
cd /mnt/bigdrive
git clone https://github.com/hec-ovi/vllm-awq4-qwen strix-qwen36-vllm-awq4-qwen
cd /mnt/bigdrive/strix-qwen36-vllm-awq4-qwen
podman build -t localhost/vllm-awq4-qwen:builder .
```

The upstream Dockerfile uses the ROCm/TheRock and patched vLLM stack needed for this hardware path.

### 3. Download models

Install Hugging Face tooling if needed:

```bash
python3 -m pip install --user -U huggingface_hub hf_xet
```

Download the AWQ4 target model:

```bash
HF_HOME=/mnt/bigdrive/strix-qwen36-vllm-awq4-qwen/models \
huggingface-cli download cyankiwi/Qwen3.6-27B-AWQ-INT4
```

Download the DFlash drafter:

```bash
huggingface-cli download z-lab/Qwen3.6-27B-DFlash \
  --local-dir /mnt/bigdrive/strix-qwen36-vllm-dflash/models/Qwen3.6-27B-DFlash \
  --local-dir-use-symlinks False
```

If either repository is gated, authenticate first:

```bash
huggingface-cli login
```

Do not commit Hugging Face tokens.

### 4. Configure environment

Copy `.env.example` and edit values if your paths differ:

```bash
cp .env.example .env
```

Minimum variables:

```bash
export API_KEY=change-me
export VLLM_AWQ4_ROOT=/mnt/bigdrive/strix-qwen36-vllm-awq4-qwen
export DFLASH_DIR=/mnt/bigdrive/strix-qwen36-vllm-dflash/models/Qwen3.6-27B-DFlash
```

### 5. Start the server

```bash
API_KEY=change-me \
VLLM_AWQ4_ROOT=/mnt/bigdrive/strix-qwen36-vllm-awq4-qwen \
DFLASH_DIR=/mnt/bigdrive/strix-qwen36-vllm-dflash/models/Qwen3.6-27B-DFlash \
./scripts/run_best_awq4_dflash_server.sh
```

The service listens on:

```text
http://0.0.0.0:18141/v1
```

It is started as a persistent Podman container with:

```text
--restart unless-stopped
```

Startup can take several minutes because vLLM profiles the full GPU/KV-cache configuration.

## Client Settings

For agents, prefer `/v1/responses` streaming:

```json
{
  "model": "Qwen3.6-27B-AWQ4",
  "input": "Write a concise Python function.",
  "stream": true,
  "temperature": 0,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Header:

```text
Authorization: Bearer change-me
```

OpenAI-compatible base URL:

```text
http://HOST:18141/v1
```

Model name:

```text
Qwen3.6-27B-AWQ4
```

## Benchmark

Run after the server is ready:

```bash
API_KEY=change-me python3 benchmarks/stream_bench.py
```

Raw local benchmark output is included in:

```text
benchmarks/stream_tuning_results.jsonl
```

## Important Findings

- `/v1/responses` with `stream:true` and `enable_thinking:false` is the correct path for agent-style usage.
- Non-streaming `/v1/responses` can put output into reasoning fields on this patched stack; avoid it for agent clients.
- `DFLASH_TOKENS=12` was slower than `8` locally.
- `MAX_NUM_SEQS=3` was required to reach 40+ tok/s aggregate.
- Very long prompts, such as 25K+ tokens, still have poor end-to-end latency even when decode is fast.
- `--enforce-eager` is intentional for this ROCm/gfx1151 path.
- `--safetensors-load-strategy eager` avoids lazy mmap-style loading behavior.

## Security

No API keys or Hugging Face tokens are committed. Replace every `change-me` value locally.

