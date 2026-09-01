#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/.github-token"

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "Error: missing $TOKEN_FILE" >&2
  exit 1
fi

TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
if [[ -z "$TOKEN" ]]; then
  echo "Error: $TOKEN_FILE is empty" >&2
  exit 1
fi

ASKPASS="$(mktemp)"
chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS"' EXIT

printf '#!/bin/sh\necho %s\n' "$TOKEN" > "$ASKPASS"

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git -c credential.helper= push "$@"
