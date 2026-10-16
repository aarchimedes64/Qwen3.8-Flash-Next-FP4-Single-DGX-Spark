# NVIDIA Qwen3.8-Flash-Next FP4 on one DGX Spark

A single-command starter for [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), using the required `eugr/spark-vllm-b12x:latest` container. It preserves NVIDIA's published DGX Spark inference profile and wraps it in a persistent Docker Compose service.

The model runs detached from SSH. Docker restarts the container after process exits and when Docker starts again, unless you deliberately stopped it. Model downloads and all five compilation/runtime caches live on the host and survive container recreation.

## Quick start

Run on the **DGX Spark**, with Linux/aarch64, the NVIDIA driver, Docker Engine, the NVIDIA Container Toolkit configured for Docker, and **Docker Compose v2.30 or newer**. Check `nvidia-smi`, `docker info`, and `docker compose version` first. Use a user with permission to run Docker; run the recipe consistently as that user so the same home-directory caches are used.

```bash
git clone https://github.com/aarchimedes64/Qwen3.8-Flash-Next-FP4-Single-DGX-Spark.git
cd Qwen3.8-Flash-Next-FP4-Single-DGX-Spark
cp .env.sample .env
chmod 600 .env
# Optional: edit .env to set HF_TOKEN and/or API_KEY.
./start.sh
```

If `.env` is missing, `start.sh` creates it with private permissions. You can also provide `HF_TOKEN` through your shell environment. Check and accept any model access terms on Hugging Face before downloading.

`./start.sh` pulls the image if absent, creates the service, and returns once the container starts. The container itself downloads the checkpoint into the persistent Hugging Face cache and prepares kernels. **Wait for the start command to finish before disconnecting**; image pulling happens before detachment. Once it returns, model download and initialization continue after SSH disconnects. First launch can take a long time; readiness time depends on the network, storage, and compilation cache.

NVIDIA lists the checkpoint at approximately **96 GiB**. Allow additional NVMe space for the image, disk-backed PLE data, and compilation caches. This recipe does not claim a measured free-disk minimum. Stop other large GPU workloads before starting.

## Everyday commands

```bash
./logs.sh                 # Follow startup/server logs; Ctrl-C only exits the viewer
./status.sh               # Starting, healthy, unhealthy, restarting, or exited
./stop.sh                 # Graceful stop, up to 120 seconds; preserves caches
./start.sh                # Start again; also applies changes made to .env
./start.sh --pull         # Pull the configured image and apply it
./start.sh --no-launch    # Validate config without Docker Engine/GPU access
```

The scripts work from another directory too when invoked by their full path. Re-running `start.sh` with unchanged configuration leaves the running service alone. After changing the image, port, token, or memory settings in `.env`, run `start.sh` again so Compose can recreate the service.

For reboot recovery, Docker Engine must start at boot. On a typical systemd-based Spark installation:

```bash
sudo systemctl enable --now docker
```

`restart: unless-stopped` leaves a deliberately stopped service stopped across reboots. Use `./start.sh` to resume it. There is no second systemd service to install. See [Docker's restart-policy behavior](https://docs.docker.com/engine/containers/start-containers-automatically/).

## API and readiness

The default OpenAI-compatible endpoint is `http://SPARK_IP:8888/v1`. Check readiness locally on the Spark:

```bash
curl --fail http://127.0.0.1:8888/health
curl --fail http://127.0.0.1:8888/v1/models
curl --fail http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-Flash-Next-FP4-Local","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

If you set `API_KEY`, add `-H "Authorization: Bearer YOUR_API_KEY"` to the `/v1` requests. The health check does not require the key. Adjust the URL if you change `BIND` or `PORT`.

The default `BIND=0.0.0.0` matches NVIDIA's recipe and exposes the API on every IPv4 interface. Set an `API_KEY` for clients on your network, or set `BIND=127.0.0.1` and use an SSH tunnel:

```bash
ssh -N -L 8888:127.0.0.1:8888 YOUR_USER@SPARK_IP
```

Then use `http://127.0.0.1:8888/v1` on your computer. `.env` is ignored by Git; avoid sharing `docker compose config` or `docker inspect` output because it can contain your tokens. `start.sh --no-launch` validates without printing those values.

## Configuration

Edit `.env`; exported shell variables take precedence over `.env`. Supported settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `IMAGE` | `eugr/spark-vllm-b12x:latest` | Required Spark container family; a digest may be used for reproducibility |
| `HF_TOKEN` | Empty | Hugging Face download token, if needed |
| `API_KEY` | Empty | Optional vLLM API authentication |
| `BIND` / `PORT` | `0.0.0.0` / `8888` | Listening address and port |
| `MAX_MODEL_LEN` | `262144` | Native context limit |
| `MAX_NUM_SEQS` | `8` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | Prefill batch budget |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM memory budget |

The recipe keeps NVIDIA's MTP-4 configuration, `modelopt_mixed` quantization, FP8 KV cache, B12X load/linear/MoE/GDN backends, disk PLE with `io_uring`, AOT compilation, Qwen reasoning/tool parsers, and all fourteen published environment settings. Full arguments are in `scripts/serve.sh`; runtime settings and mounts are in `compose.yaml`.

| Host path (default) | Container path | Override in `.env` |
| --- | --- | --- |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | `HF_CACHE` |
| `~/.cache/vllm` | `/root/.cache/vllm` | `VLLM_CACHE` |
| `~/.cache/flashinfer` | `/root/.cache/flashinfer` | `FLASHINFER_CACHE` |
| `~/.triton` | `/root/.triton` | `TRITON_CACHE` |
| `~/.cache/b12x` | `/root/.cache/b12x` | `B12X_CACHE` |

These reuse caches from NVIDIA's command when run under the same host user. Cache overrides must be absolute paths with sufficient space. Compose creates missing bind-mount directories. Keep the checkout at its current path while using it: the serving script is mounted read-only into the container.

## Troubleshooting

- **Startup is slow:** follow `./logs.sh`. The health check permits a 60-minute initialization window. A successful `/health` request marks the service healthy earlier. Check disk space and Hugging Face access if downloading stalls.
- **Unhealthy:** repeated health probes failed; Docker does not restart a container solely because it is unhealthy. Inspect the logs and fix the cause. To force a restart, run `./stop.sh` followed by `./start.sh`.
- **Restarting/exited:** inspect logs for access errors, unsupported image arguments, disk exhaustion, or GPU memory pressure. After fixing configuration, run `./start.sh` again. Restart policies do not repair configuration failures.
- **Out of memory:** stop competing GPU workloads. Try lowering `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, or `MAX_MODEL_LEN` in `.env`, then restart. The defaults reproduce NVIDIA's profile; this project has not benchmarked them or added a host-memory watchdog.
- **Port already in use:** stop the old interactive NVIDIA container/server first, or choose another `PORT`. Host networking uses the Spark's actual port directly.
- **No GPU / permission denied:** configure NVIDIA Container Toolkit for Docker and Docker group access on the Spark. This recipe targets Linux/aarch64 DGX Spark.
- **Remove the service:** run `docker compose down` from the checkout. Host caches remain; log history for the removed container does not. Logs rotate at three 20 MiB files per container.

`latest` can change. After a successful deployment, record the image digest with `docker image inspect eugr/spark-vllm-b12x:latest --format '{{index .RepoDigests 0}}'` and set `IMAGE` to that digest for repeatable deployments. Updates are explicit with `./start.sh --pull`.

## Validation and sources

```bash
python3 -m unittest discover -s tests -v
bash -n start.sh stop.sh logs.sh status.sh scripts/common.sh scripts/serve.sh
docker compose --env-file .env.sample config --quiet
```

Tests check NVIDIA argument/environment parity, Compose configuration and overrides, and lifecycle commands using mock executables. They do not pull the image, download weights, or perform GPU inference. Actual readiness and a generation request must be verified on a DGX Spark.

This starter follows the start/stop convenience of [MiaAI Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark), with a fresh Compose wrapper around [NVIDIA's DGX Spark instructions](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local#nvidia-dgx-spark) (checked October 16, 2026). NVIDIA requires the supplied community container because support for this checkpoint's compressed NVFP4 scales has not yet landed in upstream vLLM. Model and container licenses remain those of their publishers.
