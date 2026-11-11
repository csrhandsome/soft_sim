#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-all}"
if (( $# )); then shift; fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY http_proxy https_proxy all_proxy ftp_proxy
export NO_PROXY='*' no_proxy='*'
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
case "$target" in
    all) projects=(core planning robodojo) ;;
    core|planning|robodojo) projects=("$target") ;;
    *) printf 'Usage: bash scripts/sync_envs.sh [all|core|planning|robodojo] [uv sync options]\n' >&2; exit 2 ;;
esac
for project in "${projects[@]}"; do
    case "$project" in
        core) uv sync --project "$project_root" "$@" ;;
        planning) UV_PROJECT_ENVIRONMENT="$project_root/RoboTwin/.venv" uv sync --project "$project_root/envs/planning" --inexact "$@" ;;
        robodojo)
            python3 "$project_root/scripts/prepare_sources.py" --detach
            UV_PROJECT_ENVIRONMENT="$project_root/RoboDojo/.venv" uv sync --project "$project_root/envs/robodojo" --inexact "$@"
            ;;
    esac
done
