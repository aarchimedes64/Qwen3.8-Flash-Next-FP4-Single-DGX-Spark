#!/usr/bin/env bash
set -euo pipefail

# NVIDIA's DGX Spark profile; exec forwards shutdown to vLLM through Docker init.
args=(
  serve "${MODEL:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"
  --host "${BIND:-0.0.0.0}"
  --port "${PORT:-8888}"
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 16
  --load-format b12x
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}"
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}'
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}"
)
if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi
exec vllm "${args[@]}"
