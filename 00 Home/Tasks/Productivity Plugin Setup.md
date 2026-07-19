---
cssclasses:
  - dashboard
Status: Done
Category:
  - Task-Management
---
# ⚙️ Productivity Plugin Setup

## Installed and enabled

| Plugin | Role | Main use |
|---|---|---|
| Tasks | Task engine | Query/check tasks across the entire vault |
| Kanban | Visual planning | Markdown-backed task board |
| Reminder | Alerts | Notifications for markdown TODOs |
| Recent Files | Navigation | Jump back to recent work |
| Various Complements | Autocomplete | Faster typing/linking |
| Claudian Productivity Commands | Vault-specific helper | Capture task/open task center hotkeys |

## Hotkeys I set

| Hotkey | Action |
|---|---|
| `Cmd/Ctrl + Shift + T` | Capture task to [[00 Home/Tasks/Task Inbox]] |
| `Cmd/Ctrl + Shift + J` | Open [[00 Home/Tasks/Task Command Center]] |
| `Cmd/Ctrl + Shift + K` | Open [[00 Home/Tasks/Task Kanban Board]] |
| `Cmd/Ctrl + Shift + R` | Show reminders |
| `Cmd/Ctrl + Shift + E` | Open Recent Files |
| `Alt + Enter` | Edit task under cursor with Tasks |
| `Alt + K` | Toggle Kanban/Markdown view |
| `Cmd/Ctrl + Alt + Shift + R` | Refresh inferred deadlines + Kanban deadline lanes |

## Automation

- [[00 Home/Tasks/Deadline Automation]] documents the automatic deadline regeneration.

## Where to work

- Daily execution: [[00 Home/Daily Notes/2026-07-12]]
- Task overview: [[00 Home/Tasks/Task Command Center]]
- Fast capture: [[00 Home/Tasks/Task Inbox]]
- Workflow rules: [[00 Home/Tasks/Task Workflow]]

## Reminder syntax

```md
- [x] Email professor (@2026-07-12 09:30) 📅 2026-07-12 #task/admin
```

## Tasks syntax

```md
- [ ] Review paper 🔺 ⏳ 2026-07-13 📅 2026-07-15 #task/research
```

## Automation update — 2026-07-12

- Added unified task-board automation: status/property changes now regenerate [[00 Home/Tasks/Task Kanban Board]] via `.obsidian/automation/task_board_automation.py`.
- See [[00 Home/Tasks/Task Board Automation]] for triggers and status-to-lane rules.

## Plugin cleanup — 2026-07-14

- Removed unused Obsidian Git, Calendar, Full Calendar, Day Planner, Heatmap Calendar, QuickAdd, Note Toolbar, and Task Board plugins.
- Git remains available through the repository and command line; task capture remains provided by Claudian Productivity Commands.
- Tasks + Kanban are now the single task-query and visual-board stack.
