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

## Benchmark Summary

The main benchmark in this repository is the vLLM AWQ4 + DFlash path. Earlier Vulkan tests were also run on the same machine, but those used llama.cpp with GGUF models, not AWQ4+DFlash. Treat them as a backend/model-format comparison, not as an exact replacement for this vLLM setup.

### vLLM ROCm AWQ4 + DFlash

Target model: `cyankiwi/Qwen3.6-27B-AWQ-INT4`

Drafter: `z-lab/Qwen3.6-27B-DFlash`

Endpoint path: `/v1/responses`, streaming, `enable_thinking=false`

| Variant | Input tokens | Output tokens | Elapsed | Output tok/s | Notes |
|---|---:|---:|---:|---:|---|
| Single short stream | 26 | 18 | 0.851s | 21.16 | TTFT 0.03s |
| Single medium stream | 2341 | 512 | 44.853s | 11.42 | zero reasoning chars |
| Single long stream | 26071 | 512 | 409.600s | 1.25 | long prompt path remains poor |
| 3 parallel medium streams, `max_num_seqs=3` | 3 x 1499 | 1536 | 62.968s | 24.39 aggregate | 8.13-8.38 tok/s per stream |
| 3 parallel long generations, `max_num_seqs=3` | 3 x 1087 | 5924 | 134.946s | 43.90 aggregate | 14.32-15.34 tok/s per stream |
| 3 parallel long generations, DFlash `N=12` | 3 x 1087 | 6144 | 160.787s | 38.21 aggregate | worse than `N=8` |

During the best 3-stream long-generation run, vLLM engine logs showed steady-state generation windows between `47.4` and `68.0 tok/s` aggregate. Wall-clock aggregate throughput was `43.90 tok/s` after HTTP/SSE/client overhead.

### llama.cpp Vulkan RADV GGUF Comparison

These were earlier llama.cpp benchmarks on the same Strix Halo host. They are useful because they show Vulkan RADV can be very fast for GGUF, but they are not the same serving stack as vLLM AWQ4+DFlash.

Flags used: `-fa 1 -ngl 999 --no-mmap/-mmp 0 -b 512`, Vulkan RADV, usually `-ub 1024`.

| Model | Backend | Test | Prompt | Gen | Prompt tok/s | Gen tok/s |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf | llama.cpp Vulkan RADV | short | 512 | 128 | 852.96 | 47.26 |
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf | llama.cpp Vulkan RADV | medium | 4096 | 256 | 797.68 | 46.30 |
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf | llama.cpp Vulkan RADV | long | 16384 | 256 | 804.97 | 54.58 |
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf | llama.cpp Vulkan RADV | agent-like | 8192 | 512 | 863.69 | 54.37 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf | llama.cpp Vulkan RADV | short | 512 | 128 | 813.67 | 39.76 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf | llama.cpp Vulkan RADV | medium | 4096 | 256 | 803.65 | 41.71 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf | llama.cpp Vulkan RADV | long | 16384 | 256 | 799.64 | 46.15 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf | llama.cpp Vulkan RADV | agent-like | 8192 | 512 | 856.00 | 46.12 |
| Qwen3.6-27B-UD-Q6_K_XL.gguf | llama.cpp Vulkan RADV | agent-like | 8192 | 512 | 244.01 | 7.84 |

The fastest llama.cpp Vulkan result was the 35B A3B Q6 GGUF path at about `54.4-54.8 tok/s` generation. That is faster than single-stream vLLM AWQ4+DFlash, but it is a different model format, different server stack, and does not include DFlash speculative decoding or the same multimodal/tooling behavior.

### llama.cpp ROCm GGUF Comparison

For reference, upstream llama.cpp ROCm/HIP also ran on the same host:

| Model | Backend | Test | Prompt | Gen | Prompt tok/s | Gen tok/s |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf | llama.cpp ROCm | agent-like | 8192 | 512 | 843.86 | 46.09 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf | llama.cpp ROCm | agent-like | 8192 | 512 | 958.21 | 44.01 |
| Qwen3.6-27B-UD-Q6_K_XL.gguf | llama.cpp ROCm | agent-like | 8192 | 512 | 308.62 | 8.27 |

For GGUF, Vulkan RADV beat upstream ROCm/HIP in the most important 35B A3B generation tests. For AWQ4+DFlash, the tested implementation path is vLLM ROCm.

### Vulkan + DFlash Feasibility Test

Question tested: can DFlash run on Vulkan instead of ROCm?

Short answer:

- `AWQ4 + DFlash` as used here is a vLLM ROCm path. I did not find a vLLM Vulkan backend for this stack.
- A separate `GGUF target + GGUF DFlash drafter + llama.cpp Vulkan` path exists experimentally through `spiritbuun/buun-llama-cpp`, but it did not pass the local smoke test on Strix Halo/RADV.

What was tested:

- Fork: `spiritbuun/buun-llama-cpp`
- Build: `-DGGML_VULKAN=ON`
- Target: `Qwen3.6-27B-UD-Q6_K_XL.gguf`
- Drafter: `spiritbuun/Qwen3.6-27B-DFlash-GGUF`, `dflash-draft-3.6-q8_0.gguf`
- Command shape: `llama-speculative-simple -m TARGET -md DRAFT --spec-type dflash -ngl 999 -ngld 999 --no-mmap -fa on`

Result:

```text
Target model loaded on Vulkan RADV.
DFlash drafter loaded on Vulkan RADV.
Generation crashed before producing tokens:
ggml_get_n_tasks: op not implemented: SSM_CONV_TREE
fatal error in ggml_graph_plan
```

Conclusion: Vulkan+DFlash GGUF is promising enough to revisit, but it is not currently a working replacement for the ROCm AWQ4+DFlash endpoint on this machine. The stable production path remains vLLM ROCm AWQ4+DFlash.

## Repository Contents

```text
scripts/run_best_awq4_dflash_server.sh
benchmarks/stream_bench.py
benchmarks/stream_tuning_results.jsonl
benchmarks/vulkan_dflash_smoke_2026-04-29.md
reports/SPEED_TUNING_REPORT_2026-04-29.md
.env.example
```

## Reproduce The Environment

These steps assume Fedora or another Linux host with working Strix Halo ROCm device access through `/dev/kfd` and `/dev/dri`, plus Podman.

Use a large disk. The examples use `/data/qwen36-awq4-dflash`. Replace it with any large local path.

### 1. Create work directories

```bash
mkdir -p /data/qwen36-awq4-dflash
mkdir -p /data/qwen36-awq4-dflash/models
```

### 2. Clone and build the Strix Halo vLLM container

```bash
cd /data/qwen36-awq4-dflash
git clone https://github.com/hec-ovi/vllm-awq4-qwen vllm-awq4-qwen
cd /data/qwen36-awq4-dflash/vllm-awq4-qwen
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
HF_HOME=/data/qwen36-awq4-dflash/vllm-awq4-qwen/models \
huggingface-cli download cyankiwi/Qwen3.6-27B-AWQ-INT4
```

Download the DFlash drafter:

```bash
huggingface-cli download z-lab/Qwen3.6-27B-DFlash \
  --local-dir /data/qwen36-awq4-dflash/models/Qwen3.6-27B-DFlash \
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
export VLLM_AWQ4_ROOT=/data/qwen36-awq4-dflash/vllm-awq4-qwen
export DFLASH_DIR=/data/qwen36-awq4-dflash/models/Qwen3.6-27B-DFlash
```

### 5. Start the server

```bash
API_KEY=change-me \
VLLM_AWQ4_ROOT=/data/qwen36-awq4-dflash/vllm-awq4-qwen \
DFLASH_DIR=/data/qwen36-awq4-dflash/models/Qwen3.6-27B-DFlash \
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
