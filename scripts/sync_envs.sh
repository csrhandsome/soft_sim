#!/usr/bin/env bash
# Synchronize one or more isolated environments from their own lock files.
#
# Each environment is an independent uv project. The shared ``soft-sim`` package
# is installed afterwards as an editable dependency so every interpreter imports
# the same source tree without borrowing the caller's PYTHONPATH.
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
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

usage() {
    cat >&2 <<'USAGE'
Usage: bash scripts/sync_envs.sh [TARGET] [uv sync options]

TARGET:
  all       core, planning, robodojo, dataset (the default working set)
  meshing   optional TetGen environment; never part of "all"
  core | planning | robodojo | dataset
USAGE
}

case "$target" in
    all) projects=(core planning robodojo dataset) ;;
    core|planning|robodojo|dataset|meshing) projects=("$target") ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
esac

install_package() {
    local project_env="$1"
    "$uv_bin" pip install --quiet --no-deps --editable "$project_root" \
        --python "$project_env/.venv/bin/python"
}

for project in "${projects[@]}"; do
    case "$project" in
        core)
            "$uv_bin" sync --locked --project "$project_root" "$@"
            ;;
        robodojo)
            PYTHONPATH="$project_root/src" python3 -m soft_sim.sources --detach
            "$uv_bin" sync --locked --project "$project_root/envs/robodojo" "$@"
            install_package "$project_root/envs/robodojo"
            ;;
        planning|dataset|meshing)
            "$uv_bin" sync --locked --project "$project_root/envs/$project" "$@"
            install_package "$project_root/envs/$project"
            ;;
    esac
done
echo "Environments ready: ${projects[*]}"
