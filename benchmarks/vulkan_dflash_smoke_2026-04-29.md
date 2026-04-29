# Vulkan + DFlash Smoke Test

Date: 2026-04-29

This test checked whether a Vulkan path can replace the ROCm vLLM AWQ4+DFlash endpoint.

## Summary

`AWQ4 + DFlash` is currently a vLLM ROCm setup in this repository. A separate experimental `GGUF + DFlash + Vulkan` path exists through `spiritbuun/buun-llama-cpp`, but it did not complete a local smoke test on Strix Halo / RADV.

## Tested Stack

- Fork: `spiritbuun/buun-llama-cpp`
- Commit tested: `72d130efa3107b1092a748ddacd9870c9ed55d71`
- Build flag: `-DGGML_VULKAN=ON`
- Backend detected: `Vulkan0 (Radeon 8060S Graphics (RADV GFX1151))`
- Target model: `Qwen3.6-27B-UD-Q6_K_XL.gguf`
- DFlash draft model: `spiritbuun/Qwen3.6-27B-DFlash-GGUF`, file `dflash-draft-3.6-q8_0.gguf`
- Drafter SHA256: `29ba8b816eedea674e8bdabbd29db8da69539117c76da40e40d2207c0fb224db`

## Command Shape

```bash
llama-speculative-simple \
  -m /path/to/Qwen3.6-27B-UD-Q6_K_XL.gguf \
  -md /path/to/dflash-draft-3.6-q8_0.gguf \
  --spec-type dflash \
  -ngl 999 \
  -ngld 999 \
  --no-mmap \
  -fa on \
  -c 4096 \
  -cd 256 \
  -b 256 \
  -ub 64 \
  --draft-max 8 \
  --draft-min 1 \
  --draft-p-min 0.75 \
  -n 128 \
  -p "Write a Python function that computes Fibonacci iteratively. Output code only."
```

## Result

The target and drafter both loaded on Vulkan RADV, but generation crashed before producing benchmarkable output.

Relevant error:

```text
common_speculative_init: copyspec speculative decoding (gamma=6)
dflash: block_size=16, mask_token=248070, n_target_layers=5, n_embd=5120
allocate_tree_buffers: allocated tree verify buffers
ggml_get_n_tasks: op not implemented: SSM_CONV_TREE
fatal error
timeout: the monitored command dumped core
```

## Conclusion

Vulkan+DFlash GGUF is not a working replacement yet on this Strix Halo/RADV setup. The stable tested endpoint remains vLLM ROCm AWQ4+DFlash.
