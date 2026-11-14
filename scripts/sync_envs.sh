#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
fi
target="${1:-all}"
if (( $# )); then shift; fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY http_proxy https_proxy all_proxy ftp_proxy
export NO_PROXY='*' no_proxy='*'
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
case "$target" in
    all) projects=(core planning robodojo dataset) ;;
    core|planning|robodojo|dataset) projects=("$target") ;;
    *) printf 'Usage: bash scripts/sync_envs.sh [all|core|planning|robodojo|dataset] [uv sync options]\n' >&2; exit 2 ;;
esac
for project in "${projects[@]}"; do
    case "$project" in
        core) "$uv_bin" sync --locked --project "$project_root" "$@" ;;
        planning) "$uv_bin" sync --locked --project "$project_root/envs/planning" "$@" ;;
        dataset) "$uv_bin" sync --locked --project "$project_root/envs/dataset" "$@" ;;
        robodojo)
            python3 "$project_root/scripts/prepare_sources.py" --detach
            "$uv_bin" sync --locked --project "$project_root/envs/robodojo" "$@"
            ;;
    esac
done
