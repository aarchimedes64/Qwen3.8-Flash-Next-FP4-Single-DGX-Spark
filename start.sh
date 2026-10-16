#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"

case "${1:-}" in
  '') mode=start ;;
  --no-launch) mode=validate ;;
  --pull) mode=pull ;;
  *) echo 'Usage: ./start.sh [--no-launch | --pull]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments.' >&2; exit 2; }

if [[ ! -f "$RECIPE_DIR/.env" ]]; then
  (umask 077; cp "$RECIPE_DIR/.env.sample" "$RECIPE_DIR/.env")
  echo 'Created .env from .env.sample; edit it to set HF_TOKEN or API_KEY if needed.'
fi
compose config --quiet
if [[ "$mode" == validate ]]; then
  echo 'Compose configuration is valid. No image pulled or container started. Secrets are hidden.'
  echo 'Launch command: docker compose up --detach vllm (from this repository).'
  exit 0
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
  echo 'Run this recipe on the Linux/aarch64 DGX Spark. Use --no-launch for local configuration validation.' >&2
  exit 1
}
docker info >/dev/null
if [[ "$mode" == pull ]]; then
  compose pull vllm
fi
compose up --detach vllm
echo 'vLLM is starting in the background. You can disconnect SSH now.'
echo 'First launch downloads the model and compiles kernels; check ./logs.sh and ./status.sh for readiness.'
