# Qwen3.6-27B AWQ4 + DFlash Speed Tuning Report

Date: 2026-04-29
Machine: AMD Strix Halo / Ryzen AI Max+ 395 / Radeon 8060S gfx1151 / 128 GB UMA
Endpoint: `http://0.0.0.0:18141/v1`
Model: `Qwen3.6-27B-AWQ4`

## Sources checked

- https://github.com/hec-ovi/vllm-awq4-qwen
- https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4
- https://huggingface.co/z-lab/Qwen3.6-27B-DFlash
- https://github.com/vllm-project/vllm/pull/40898

Key findings from current upstream/repo material:

- The Strix Halo-specific repo targets exactly this hardware and model path: AWQ4 target + DFlash drafter, ROCm/TheRock, patched vLLM.
- Recommended agent path is `/v1/responses` with `stream:true` and `chat_template_kwargs.enable_thinking=false`.
- For tool agents, avoid non-streaming `/v1/responses` with `enable_thinking=false`; streaming is the patched path.
- `--enforce-eager` is intentional on gfx1151 because graph paths are unstable.
- DFlash `N=8` is the repo daily-driver value. Larger speculation is not automatically faster.
- 40-50 tok/s is realistic as aggregate throughput with concurrent streams, not as one single Hermes request.

## Local tested variants

### Baseline stable, max_num_seqs=1, DFlash=8

Streaming `/v1/responses`, `enable_thinking=false`:

| test | input tok | output tok | elapsed | wall output t/s | notes |
|---|---:|---:|---:|---:|---|
| short stream | 26 | 18 | 0.851s | 21.16 | TTFT 0.03s |
| medium stream | 2341 | 512 | 44.853s | 11.42 | zero reasoning |
| long stream | 26071 | 512 | 409.600s | 1.25 | long-context wall remains poor |
| 3 parallel medium | 3x1499 | 1536 | 100.151s | 15.34 aggregate | queued by max_num_seqs=1 |

### Tuned concurrency, max_num_seqs=3, DFlash=8

This is the best tested variant.

| test | input tok each | output tok total | elapsed | aggregate output t/s | per stream t/s |
|---|---:|---:|---:|---:|---|
| 3 parallel medium | 1499 | 1536 | 62.968s | 24.39 | 8.13-8.38 |
| 4 parallel medium | 1499 | 2048 | 103.891s | 19.71 | 4.93-7.40 |
| 3 parallel long generation | 1087 | 5924 | 134.946s | 43.90 | 14.32-15.34 |

Engine log windows during 3-stream long generation:

- 47.4 tok/s
- 42.4 tok/s
- 62.6 tok/s
- 61.5 tok/s
- 51.1 tok/s
- 48.7 tok/s
- 54.2 tok/s
- 52.2 tok/s
- 68.0 tok/s
- 55.2 tok/s

Observed DFlash acceptance during good windows: about 39-60%.

### DFlash=12, max_num_seqs=3

Worse than N=8.

| test | output tok total | elapsed | aggregate output t/s | per stream t/s |
|---|---:|---:|---:|---|
| 3 parallel long generation | 6144 | 160.787s | 38.21 | 12.74-13.23 |

DFlash=12 increased draft work and lowered effective acceptance. Rejected.

## Final selected launcher

Path:

`/data/qwen36-awq4-dflash/vllm-awq4-qwen/run_best_awq4_dflash_server.sh`

Defaults now set to:

- `MAX_MODEL_LEN=262144`
- `MAX_NUM_SEQS=3`
- `DFLASH_TOKENS=8`
- `MAX_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTIL=0.90`
- `--safetensors-load-strategy eager`
- `/v1/responses` streaming recommended for Hermes

## Practical recommendation for Hermes

Use `/v1/responses` streaming if Hermes supports it. Send:

```json
{
  "model": "Qwen3.6-27B-AWQ4",
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Expected realistic throughput:

- Single active request: ~14-21 tok/s depending prompt/output shape.
- Three active long-generation streams: ~44 tok/s aggregate wall-clock.
- Engine steady-state windows can hit 50-68 tok/s aggregate.
- Very long prompts, e.g. 25K+ tokens, still have poor end-to-end latency even when decode is fast.

## Endpoint credentials

Base URL wired/LAN:

`http://YOUR_HOST_OR_IP:18141/v1`

Base URL Wi-Fi/LAN:

`http://YOUR_HOST_OR_IP:18141/v1`

Model:

`Qwen3.6-27B-AWQ4`

API key:

`change-me`

Header:

`Authorization: Bearer change-me`

## Current status

Final server restarted and smoke-tested after restoring DFlash=8. `/v1/models` returns 200 and streaming `/v1/responses` returns SSE output.
