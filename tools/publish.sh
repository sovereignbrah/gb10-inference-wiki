#!/bin/bash
# The only publish path: scrub in strict mode, then push. Never push directly.
set -e
cd "$(dirname "$0")/.."
PUBLIC_LAB_STRICT=1 ./tools/scrub.sh
git push origin main
echo "published."
