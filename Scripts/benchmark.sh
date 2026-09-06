#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
app_binary="${ASTRA_DERIVED_DATA:-/tmp/MLXAstraDerived}/Build/Products/Release/MLXAstra.app/Contents/MacOS/MLXAstra"
if [[ ! -x "$app_binary" ]]; then
  bash Scripts/build.sh
fi
"$app_binary" --benchmark "$@"
