#!/bin/sh
# Deploy the working tree to Railway and wait until the new build answers on /healthz.
set -e
cd "$(dirname "$0")/.."
VERSION=$(git rev-parse --short HEAD)
echo "$VERSION" > VERSION
railway up --detach --json </dev/null >/dev/null
URL=$(railway domain --json </dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['domain'])")
echo "deploying $VERSION to $URL"
for i in $(seq 1 90); do
  live=$(curl -s --max-time 10 "$URL/healthz" | python3 -c "import sys,json; print(json.load(sys.stdin).get('build'))" 2>/dev/null || true)
  if [ "$live" = "$VERSION" ]; then echo "live after ${i}0s"; exit 0; fi
  sleep 10
done
echo "timed out waiting for $VERSION (live build: $live)"; exit 1
