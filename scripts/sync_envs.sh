#!/usr/bin/env bash
# Sync independent projects with proxy variables removed from the child process.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-all}"
if (( $# > 0 )); then
    shift
fi

case "$target" in
    all) projects=(isaacsim mujoco) ;;
    isaacsim|mujoco) projects=("$target") ;;
    *) printf 'Usage: bash scripts/sync_envs.sh [all|isaacsim|mujoco] [uv sync options]\n' >&2; exit 2 ;;
esac

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
unset http_proxy https_proxy all_proxy ftp_proxy
export NO_PROXY='*'
export no_proxy='*'
# Each project must use its own .venv even when called from an activated shell.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

# Isaac Sim wheels are large; allow slow reads while limiting bandwidth contention.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"

failed=0
for project in "${projects[@]}"; do
    printf 'Syncing %s with proxies disabled...\n' "$project"
    if (cd -- "$project_root/$project" && uv sync "$@"); then
        printf '%s: sync complete.\n' "$project"
    else
        printf '%s: sync failed.\n' "$project" >&2
        failed=1
    fi
done
exit "$failed"
