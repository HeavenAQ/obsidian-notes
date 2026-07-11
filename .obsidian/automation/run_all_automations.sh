#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
TODAY="${1:-$(date +%F)}"
python3 .obsidian/automation/task_board_automation.py --today "$TODAY"
