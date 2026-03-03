#!/usr/bin/env bash
# Monitors a GitHub PR for new Codex review comments.
# Usage: ./scripts/watch-pr-review.sh [PR_NUMBER] [INTERVAL_SECONDS]

set -euo pipefail

PR=${1:-1}
INTERVAL=${2:-30}
SEEN_COUNT=0

echo "👁  Watching PR #${PR} for Codex review (polling every ${INTERVAL}s) ..."
echo "    Press Ctrl+C to stop."
echo ""

while true; do
  COMMENTS=$(gh pr view "$PR" --comments --json comments -q '.comments[] | select(.author.login == "chatgpt-codex-connector") | .body' 2>/dev/null || echo "")
  COUNT=$(echo "$COMMENTS" | grep -c "." 2>/dev/null; true)
  COUNT=${COUNT:-0}

  if [[ "$COUNT" -gt "$SEEN_COUNT" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🆕 Новый комментарий от Codex ($(date '+%H:%M:%S')):"
    echo ""
    echo "$COMMENTS" | tail -n +"$((SEEN_COUNT + 1))"
    echo ""

    # Check for issues
    if echo "$COMMENTS" | grep -qiE "suggestion|issue|problem|fix|error|should|consider|missing|incorrect|bug"; then
      echo "⚠️  Codex нашёл замечания — нужны исправления."
    else
      echo "✅ Codex одобрил без замечаний."
    fi

    SEEN_COUNT=$COUNT
  fi

  sleep "$INTERVAL"
done
