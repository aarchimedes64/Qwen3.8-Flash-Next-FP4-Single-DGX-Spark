#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

command -v docker >/dev/null 2>&1 || { echo 'Install Docker Engine and Docker Compose v2.30+ first.' >&2; exit 1; }
docker compose version >/dev/null

# Resolve configuration and bind mounts from the repository, even from another cwd.
compose() {
  docker compose --project-directory "$RECIPE_DIR" --env-file "$RECIPE_DIR/.env" -f "$RECIPE_DIR/compose.yaml" "$@"
}
