#!/usr/bin/env bash
# Run the promptfoo evaluation suite against the TOI RAG API.
#
# Prerequisites:
#   - TOI RAG API is running (uvicorn app.main:app --reload)
#   - Node.js installed (npx available)
#   - httpx installed (pip install httpx)
#
# Environment variables:
#   TOI_BASE_URL        API base URL (default: http://localhost:8000)
#   TOI_EVAL_EMAIL      Login email for the eval user
#   TOI_EVAL_PASSWORD   Login password for the eval user
#   OPENAI_API_KEY      Required by promptfoo for llm-rubric / graded assertions
#
# Usage:
#   bash eval/scripts/run_eval.sh
#
# Multi-model comparison:
#   bash eval/scripts/run_eval.sh --providers eval/promptfoo.providers.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_DIR="$REPO_ROOT/eval"
REPORTS_DIR="$EVAL_DIR/reports"

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

echo "==> Obtaining session cookie..."
COOKIE=$(python "$EVAL_DIR/scripts/get_session_cookie.py")
export TOI_SESSION_COOKIE="$COOKIE"
echo "    Session cookie obtained."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$REPORTS_DIR/results_${TIMESTAMP}.json"

echo "==> Running promptfoo eval..."
npx promptfoo eval \
  --config "$EVAL_DIR/promptfoo.yaml" \
  --output "$OUTPUT_FILE" \
  "$@"

echo ""
echo "==> Results saved to: $OUTPUT_FILE"
echo "==> Opening interactive report..."
npx promptfoo view
