---
cssclasses:
  - dashboard
Status: Done
Category:
  - Task-Management
---
# 🔁 Deadline Automation

Deadline automation is now part of the unified [[00 Home/Tasks/Task Board Automation]] pipeline.

## What it updates

- [[00 Home/Tasks/Deadline Triage]]
- Deadline lanes inside [[00 Home/Tasks/Task Kanban Board]]

## Automation triggers

- Obsidian startup / layout ready
- Every 30 minutes while Obsidian is open
- 60 seconds after markdown files are created, modified, or deleted
- Manual command: `Claudian Productivity Commands: Refresh automations: deadlines + task board`
- Hotkey: `Cmd/Ctrl + Alt + Shift + R`

## Current deadline rule

Research `Assigned Date` items are intentionally excluded from urgent **Overdue** and **Today** deadline lanes. Research pages still appear in generated page-status lanes, so reading progress remains visible without polluting the immediate deadline queue.

## Generator scripts

```text
.obsidian/automation/task_board_automation.py
.obsidian/automation/deadline_triage.py
```
