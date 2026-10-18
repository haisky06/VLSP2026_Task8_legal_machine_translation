#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <host:port|url> <input.md> <output.md> [auto|en-vi|vi-en]" >&2
  exit 2
fi

HOST="$1"
INPUT="$2"
OUTPUT="$3"
DIRECTION="${4:-auto}"

if [[ ! -f "$INPUT" ]]; then
  echo "Input file does not exist: $INPUT" >&2
  exit 2
fi
case "$DIRECTION" in
  auto|en-vi|vi-en) ;;
  *) echo "Invalid direction: $DIRECTION" >&2; exit 2 ;;
esac

if [[ "$HOST" =~ ^https?:// ]]; then
  BASE_URL="${HOST%/}"
else
  BASE_URL="http://${HOST%/}"
fi

mkdir -p "$(dirname "$OUTPUT")"
TMP_OUTPUT="$(mktemp "${OUTPUT}.tmp.XXXXXX")"
TMP_HEADERS="$(mktemp)"
trap 'rm -f "$TMP_OUTPUT" "$TMP_HEADERS"' EXIT

curl --fail --show-error --silent \
  --connect-timeout 30 \
  --max-time 86400 \
  --dump-header "$TMP_HEADERS" \
  --form "direction=$DIRECTION" \
  --form "file=@${INPUT};type=text/markdown" \
  "${BASE_URL}/translate" \
  --output "$TMP_OUTPUT"

if [[ ! -s "$TMP_OUTPUT" ]]; then
  echo "Translation is empty" >&2
  exit 1
fi
mv -f "$TMP_OUTPUT" "$OUTPUT"
grep -i '^X-VLSP-' "$TMP_HEADERS" | tr -d '\r' >&2 || true
echo "Saved UTF-8 Markdown translation to: $OUTPUT" >&2

