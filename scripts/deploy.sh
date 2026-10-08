#!/bin/sh
# Deploy the working tree to Railway and wait until the new build answers on /healthz.
set -e
cd "$(dirname "$0")/.."
VERSION=$(git rev-parse --short HEAD)
echo "$VERSION" > VERSION
railway up --detach --json </dev/null >/dev/null
URL=$(railway domain --json </dev/null 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('domain') or d['domains'][0]['domain'])" 2>/dev/null || true)
URL=${URL:-${RAILWAY_URL:-https://ttb-label-verifier-production-28e6.up.railway.app}}
case "$URL" in http*) ;; *) URL="https://$URL" ;; esac
echo "deploying $VERSION to $URL"
for i in $(seq 1 90); do
  live=$(curl -s --max-time 10 "$URL/healthz" | python3 -c "import sys,json; print(json.load(sys.stdin).get('build'))" 2>/dev/null || true)
  if [ "$live" = "$VERSION" ]; then echo "live after ${i}0s"; exit 0; fi
  sleep 10
done
echo "timed out waiting for $VERSION (live build: $live)"; exit 1
