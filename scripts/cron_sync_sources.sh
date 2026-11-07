#!/usr/bin/env bash
# ==============================================================================
# cron_sync_sources.sh
# Shell wrapper for overnight mutual fund data sync and re-indexing.
# Configured to run daily via Linux crontab: 0 2 * * *
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR"
TIMESTAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
LOG_FILE="$LOGS_DIR/overnight_sync.log"

echo "=== Overnight Sync Triggered at $(date) ===" >> "$LOG_FILE"

cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

python3 "$PROJECT_DIR/scripts/overnight_update.py" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Overnight Sync Completed Successfully at $(date) ===" >> "$LOG_FILE"
else
    echo "=== Overnight Sync Failed with exit code $EXIT_CODE at $(date) ===" >> "$LOG_FILE"
fi

exit $EXIT_CODE
