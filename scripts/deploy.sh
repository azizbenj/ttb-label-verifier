#!/bin/sh
# Deploy the working tree to Railway and wait until that deployment answers on /healthz.
set -e
cd "$(dirname "$0")/.."
URL=${RAILWAY_URL:-https://ttb-label-verifier-production-28e6.up.railway.app}
for attempt in 1 2 3; do
  OUT=$(railway up --detach --json </dev/null 2>&1) && break
  echo "railway up failed (attempt $attempt): $OUT"; sleep 5
done
ID=$(printf '%s' "$OUT" | python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip().splitlines()[-1])['deploymentId'])")
echo "deployment $ID uploaded; waiting for it on $URL"
for i in $(seq 1 90); do
  live=$(curl -s --max-time 10 "$URL/healthz" | python3 -c "import sys,json; print(json.load(sys.stdin).get('deployment'))" 2>/dev/null || true)
  if [ "$live" = "$ID" ]; then echo "live after ${i}0s"; exit 0; fi
  sleep 10
done
echo "timed out (live deployment: $live)"; exit 1
