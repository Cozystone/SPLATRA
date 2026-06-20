#!/usr/bin/env bash
# Launch the plugin API on :8000. Requires API extras: pip install -e ".[api]"
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn apps.plugin_api:app --host 0.0.0.0 --port 8000 "$@"
