---
cssclasses:
  - dashboard
obsidianUIMode: preview
Status: In-Progress
Category:
  - Task-Management
---
# ✅ Task Command Center

> [!tip] Operating rule
> Capture anywhere, then process here. Use dates only when a task truly needs scheduling; otherwise keep it in Inbox/Next so it does not pollute the calendar.

## ⚡ Quick links

- 📥 [[00 Home/Tasks/Task Inbox|Task Inbox]]
- 🧨 [[00 Home/Tasks/Deadline Triage|Deadline Triage]]
- 📅 [[00 Home/Daily Notes/2026-07-12|Today’s daily note]]
- 🏠 [[00 Home/00 Home|Dashboard]]
- 🎓 [[00 Home/Learning Hub.base|Learning Hub]]
- 🔬 [[02 Research/Research Hub.base|Research Hub]]
- 🧩 [[03 Algorithms/Algorithms Hub.base|Algorithms Hub]]

## 🧨 Inferred deadlines / hidden backlog

```tasks
not done
path includes 00 Home/Tasks/Deadline Triage
due before tomorrow
sort by due
sort by description
hide task count
```

## 🚨 Overdue

```tasks
not done
due before today
sort by due
sort by priority
hide task count
```

## 📅 Due today

```tasks
not done
due today
sort by priority
sort by path
hide task count
```

## ⏳ Scheduled today

```tasks
not done
scheduled today
sort by priority
sort by path
hide task count
```

## 🗓️ Next 7 days

```tasks
not done
due after today
due before in 8 days
sort by due
sort by priority
hide task count
```

## 📥 Needs triage / no date

```tasks
not done
no due date
no scheduled date
no start date
path does not include 99 Assets
path does not include .obsidian
sort by path
hide task count
limit 80
```

## 🔬 Research tasks

```tasks
not done
path includes 02 Research
sort by priority
sort by due
hide task count
```

## 🎓 Learning / homework tasks

```tasks
not done
filter by function ['01.1 Computer Vision — Foundations Study Tracker', '01.2 DL Daily Lessons — Step-by-Step + Quiz', '01.3 DL Homework Practice — MIT 6.7960'].some(folder => task.file.path.startsWith(folder + '/'))
sort by due
sort by priority
hide task count
limit 80
```

## 🧩 Algorithms tasks

```tasks
not done
path includes 03 Algorithms
sort by priority
sort by due
hide task count
```

## 📚 Reference / docs tasks

```tasks
not done
path includes 04 Reference
sort by priority
sort by due
hide task count
```

## ✅ Recently completed

```tasks
done
done after one week ago
sort by done reverse
hide task count
limit 30
```

## 🧭 Task conventions

- `#task/inbox` — unprocessed capture
- `#task/research` — thesis/research work
- `#task/homework` — learning/homework deliverables
- `#task/admin` — logistics, emails, setup
- `#task/waiting` — blocked by someone/something
- Use `📅` only for true deadlines.
- Use `⏳` for the day you intend to work on something.
- Use Reminder syntax `(@YYYY-MM-DD HH:mm)` only for tasks that need an alert.
