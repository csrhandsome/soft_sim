#!/usr/bin/env bash
# Keep upstream benchmarks separate; initialize only simulation dependencies by default.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target=all
dry_run=0
with_policies=0
for arg in "$@"; do
    case "$arg" in
        all|robotwin|robodojo) target="$arg" ;;
        --direct)
            unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
            unset http_proxy https_proxy all_proxy ftp_proxy
            export NO_PROXY='*' no_proxy='*'
            ;;
        --dry-run) dry_run=1 ;;
        --with-policies) with_policies=1 ;;
        *) printf 'Usage: bash scripts/clone_benchmarks.sh [all|robotwin|robodojo] [--direct] [--dry-run] [--with-policies]\n' >&2; exit 2 ;;
    esac
done
export GIT_TERMINAL_PROMPT=0
failed=0
for name in RoboTwin RoboDojo; do
    if [[ "$target" != all && "${name,,}" != "$target" ]]; then continue; fi
    case "$name" in
        RoboTwin) url=https://github.com/RoboTwin-Platform/RoboTwin.git ;;
        RoboDojo) url=https://github.com/RoboDojo-Benchmark/RoboDojo.git ;;
    esac
    dest="$project_root/$name"
    submodules=()
    if [[ "$name" == RoboDojo ]]; then
        submodules+=(third_party/IsaacLab third_party/curobo)
    fi
    if (( with_policies )); then submodules+=(XPolicyLab); fi
    if (( dry_run )); then
        printf '%s -> %s (submodules: %s; no assets or package installation)\n' \
            "$url" "$dest" "${submodules[*]:-none}"
        continue
    fi
    if [[ -L "$dest" ]]; then
        printf '%s: refusing a symlink destination\n' "$name" >&2
        failed=1
        continue
    elif [[ -e "$dest/.git" ]]; then
        origin="$(git -C "$dest" remote get-url origin)"
        if [[ "$origin" != "$url" ]]; then
            printf '%s: unexpected origin: %s\n' "$name" "$origin" >&2
            failed=1
            continue
        fi
    elif ! git clone --no-recurse-submodules "$url" "$dest"; then
        failed=1
        continue
    fi
    # Preserve the checkout revision and local changes. Never pull/reset.
    # Explicit paths keep optional policy repositories from appearing again.
    if (( ${#submodules[@]} )); then
        if ! git -C "$dest" submodule update --init --recursive -- "${submodules[@]}"; then
            failed=1
        fi
    fi
done
exit "$failed"
