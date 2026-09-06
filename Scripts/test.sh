#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
xcodebuild -project MLXAstra.xcodeproj -scheme MLXAstra \
  -configuration Release -destination 'platform=macOS,arch=arm64' \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -derivedDataPath "${ASTRA_DERIVED_DATA:-/tmp/MLXAstraDerived}" ARCHS=arm64 ONLY_ACTIVE_ARCH=YES test "$@"
