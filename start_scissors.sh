#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys' >/dev/null 2>&1; then
    exec python3 "$script_dir/scripts/launch.py" "$@"
elif command -v python >/dev/null 2>&1 && python -c 'import sys' >/dev/null 2>&1; then
    exec python "$script_dir/scripts/launch.py" "$@"
else
    printf '%s\n' 'Scissors requires Python 3.12 or newer on PATH.' >&2
    exit 1
fi
