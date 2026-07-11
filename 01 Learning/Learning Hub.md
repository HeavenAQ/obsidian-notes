---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🎓 Learning Hub

## Collections

- 👁️ **[[01 Learning/Computer Vision — Foundations Study Tracker/Computer Vision — Foundations Study Tracker.base|Computer Vision Foundations]]**
- 🧠 **[[01 Learning/DL Daily Lessons — Step-by-Step + Quiz/DL Daily Lessons — Step-by-Step + Quiz.base|DL Daily Lessons]]**
- 📝 **[[01 Learning/DL Homework Practice — MIT 6.7960/DL Homework Practice — MIT 6.7960.base|MIT 6.7960 Homework]]**

## Up next

```dataviewjs
const lessons = dv.pages('"01 Learning/DL Daily Lessons — Step-by-Step + Quiz"')
  .where(p => p.Studied !== true || p["Quiz taken"] !== true)
  .sort(p => p.Date ?? "9999-12-31", 'asc')
  .limit(8)
  .map(p => [p.file.link, p.Date ?? "—", p.Studied ?? false, p["Quiz taken"] ?? false]);

dv.table(["Lesson", "Date", "Studied", "Quiz taken"], lessons);
```

```dataviewjs
const homework = dv.pages('"01 Learning/DL Homework Practice — MIT 6.7960"')
  .where(p => p.Status !== "Done")
  .sort(p => p["Due Date"] ?? "9999-12-31", 'asc')
  .limit(8)
  .map(p => [p.file.link, p.Status ?? "—", p["Due Date"] ?? "—", p["Notes ready"] ?? false]);

dv.table(["Homework", "Status", "Due", "Notes ready"], homework);
```

```dataviewjs
const cv = dv.pages('"01 Learning/Computer Vision — Foundations Study Tracker"')
  .where(p => p.Status !== "Done")
  .sort(p => p.Day ?? 999, 'asc')
  .limit(8)
  .map(p => [p.file.link, p.Status ?? "—", p.Day ?? "—", p.Confidence ?? "—"]);

dv.table(["CV topic", "Status", "Day", "Confidence"], cv);
```
