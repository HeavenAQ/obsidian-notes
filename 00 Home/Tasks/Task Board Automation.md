---
cssclasses:
  - dashboard
Status: Done
Category:
  - Task-Management
---
# 🤖 Task Board Automation

This vault now has one unified automation pipeline for the task board.

## What updates automatically

- [[00 Home/Tasks/Task Kanban Board]]
- [[00 Home/Tasks/Deadline Triage]]

## Status → Kanban behavior

When you change a page property in an Obsidian/Notion-style page, the card moves automatically on [[00 Home/Tasks/Task Kanban Board]]:

| Page property | Generated Kanban lane |
|---|---|
| `Status: Not-Started` / `Status: To-Solve` / `Reading Status: To-Read` | 📌 Page To Do / Not Started |
| `Status: In-Progress` / `Reading Status: Reading` / `Studied: true` | 🚧 Page In Progress / Reading |
| `Status: To-Review` / `Reading Status: Skim-Skip` | 🔁 Page Review / Skim |
| `Status: Done` / `Reading Status: Read` / `Studied: true` + `Quiz taken: true` | ✅ Page Done / Read |

## Automation triggers

- Obsidian startup / layout ready
- Every 30 minutes while Obsidian is open
- 60 seconds after any markdown page is created, modified, or deleted
- Manual command: `Claudian Productivity Commands: Refresh automations: deadlines + task board`
- Hotkey: `Cmd/Ctrl + Alt + Shift + R`

## Source scripts

```text
.obsidian/automation/task_board_automation.py   # unified board + deadline runner
.obsidian/automation/status_board.py            # status/property scanner
.obsidian/automation/deadline_triage.py         # inferred deadline scanner
.obsidian/automation/run_all_automations.sh     # terminal wrapper
```

## Important rule

Generated lanes are automation-owned. To move a generated page card, edit the page's source property instead of dragging the generated card manually.

Manual lanes are preserved:

- 📥 Inbox
- 🎯 Next
- 🚧 Doing
- 🧱 Waiting / Blocked
- ✅ Manual Done
- Archive
