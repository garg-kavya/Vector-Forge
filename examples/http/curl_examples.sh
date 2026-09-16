#!/usr/bin/env bash
# VectorForge HTTP API walkthrough (docs/http-api.md). Also used as the Docker smoke test.
#
#   vectorforge serve --data-dir ./data &
#   bash examples/http/curl_examples.sh [base-url] [api-key]
#
# Leaves a collection "docs" with a snapshot behind. Fails on the first unexpected response.
set -euo pipefail

BASE="${1:-http://127.0.0.1:8080}"
KEY="${2:-}"
AUTH=()
if [[ -n "$KEY" ]]; then
  AUTH=(-H "Authorization: Bearer $KEY")
fi

call() {  # call METHOD PATH [JSON-BODY]
  local method="$1" path="$2"
  shift 2
  echo "> $method $path" >&2
  if [[ $# -gt 0 ]]; then
    curl -fsS -X "$method" "${AUTH[@]}" -H 'Content-Type: application/json' -d "$1" "$BASE$path"
  else
    curl -fsS -X "$method" "${AUTH[@]}" "$BASE$path"
  fi
  echo
}

call GET /healthz
call GET /v1/status

# Start from a clean collection (a missing one answers 404, which is fine here).
curl -sS -o /dev/null -X DELETE "${AUTH[@]}" "$BASE/v1/collections/docs" || true
sleep 0.2

call POST /v1/collections '{
  "name": "docs", "dim": 4, "metric": "cosine",
  "index": {"type": "hnsw", "M": 16, "ef_construction": 200, "ef_search": 64, "seed": 42}
}'

call POST /v1/collections/docs/vectors '{
  "vectors": [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
    {"id": 2, "vector": [0.4, 0.3, 0.2, 0.1]},
    {"id": 3, "vector": [0.1, 0.1, 0.9, 0.1]}
  ]
}'

# Binary bulk insert: "VFB1", u32 dim, u64 count, count x u64 ids, count x dim x f32 (little-endian).
# Ids 10 and 11; vectors [1, 0, 0, 0] and [0, 1, 0, 0].
printf 'VFB1\x04\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00' > /tmp/vf_bulk.bin
printf '\x0a\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00' >> /tmp/vf_bulk.bin
printf '\x00\x00\x80\x3f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' >> /tmp/vf_bulk.bin
printf '\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x00\x00\x00\x00\x00\x00' >> /tmp/vf_bulk.bin
echo "> POST /v1/collections/docs/vectors:bulk" >&2
curl -fsS "${AUTH[@]}" -H 'Content-Type: application/octet-stream' \
  --data-binary @/tmp/vf_bulk.bin "$BASE/v1/collections/docs/vectors:bulk"
echo

result=$(call POST /v1/collections/docs/search '{"vector": [0.1, 0.2, 0.3, 0.41], "k": 2, "ef_search": 32}')
echo "$result"
grep -q '"results": \[{"distance":[^}]*,"id":1}' <<<"$result" || { echo "expected id 1 first" >&2; exit 1; }

call POST /v1/collections/docs/search:batch '{"vectors": [[1, 0, 0, 0], [0, 1, 0, 0]], "k": 1}'
call GET /v1/collections/docs/vectors/3
call DELETE /v1/collections/docs/vectors/2
call GET /v1/collections/docs/stats
call POST /v1/collections/docs/compact
call POST /v1/collections/docs/snapshot
call GET /v1/collections

# Errors are JSON with a stable code.
status=$(curl -sS -o /tmp/vf_err.json -w '%{http_code}' "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d '{"vector": [1, 2]}' "$BASE/v1/collections/docs/search")
cat /tmp/vf_err.json
echo
[[ "$status" == 400 ]] || { echo "expected 400, got $status" >&2; exit 1; }
grep -q DIMENSION_MISMATCH /tmp/vf_err.json
echo "all examples succeeded" >&2
