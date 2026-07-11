---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🎓 Learning Hub

> [!important] Primary hub
> This hub is now maintained as a Base: [[01 Learning/Learning Hub.base|Learning Hub Base]].

## Collections

- 👁️ **[[01 Learning/Computer Vision — Foundations Study Tracker/Computer Vision — Foundations Study Tracker.base|Computer Vision Foundations Base]]**
- 🧠 **[[01 Learning/DL Daily Lessons — Step-by-Step + Quiz/DL Daily Lessons — Step-by-Step + Quiz.base|DL Daily Lessons Base]]**
- 📝 **[[01 Learning/DL Homework Practice — MIT 6.7960/DL Homework Practice — MIT 6.7960.base|MIT 6.7960 Homework Base]]**

## 📊 Learning Progress Summary

```dataviewjs
const rows = [
  {
    area: "👁️ Computer Vision",
    pages: dv.pages('"01 Learning/Computer Vision — Foundations Study Tracker"').where(p => p.file.ext === "md"),
    done: p => p.Status === "Done",
    active: p => p.Status === "In progress",
    todo: p => p.Status === "Not started"
  },
  {
    area: "🧠 DL Lessons",
    pages: dv.pages('"01 Learning/DL Daily Lessons — Step-by-Step + Quiz"').where(p => p.file.ext === "md"),
    done: p => p.Studied === true && p["Quiz taken"] === true,
    active: p => p.Studied === true && p["Quiz taken"] !== true,
    todo: p => p.Studied !== true
  },
  {
    area: "📝 DL Homework",
    pages: dv.pages('"01 Learning/DL Homework Practice — MIT 6.7960"').where(p => p.file.ext === "md"),
    done: p => p.Status === "Done",
    active: p => p.Status === "In progress",
    todo: p => p.Status === "Not started"
  }
].map(c => {
  const total = c.pages.length;
  const done = c.pages.where(c.done).length;
  const active = c.pages.where(c.active).length;
  const todo = c.pages.where(c.todo).length;
  const pct = total ? Math.round(done / total * 100) : 0;
  const bar = "█".repeat(Math.round(pct / 10)) + "░".repeat(10 - Math.round(pct / 10));
  return [c.area, total, `✅ ${done}`, `🟡 ${active}`, `⬜ ${todo}`, `\`${bar}\` ${pct}%`];
});

dv.table(["Collection", "Total", "Done", "Active", "To Do", "Progress"], rows);
```

## 👁️ All Computer Vision Topics

```dataviewjs
const rows = dv.pages('"01 Learning/Computer Vision — Foundations Study Tracker"')
  .where(p => p.file.ext === "md")
  .sort(p => p.Day ?? 999, 'asc')
  .map(p => [
    p.file.link,
    p.Status ?? "—",
    p.Day ?? "—",
    p.Part ?? "—",
    p.Chapters ?? "—",
    p["Reading done"] ?? false,
    p["Self-check done"] ?? false,
    p.Confidence ?? "—",
    p.Date ?? "—"
  ]);

dv.table(["Topic", "Status", "Day", "Part", "Chapters", "Reading", "Self-check", "Confidence", "Date"], rows);
```

## 🧠 All DL Daily Lessons

```dataviewjs
const rows = dv.pages('"01 Learning/DL Daily Lessons — Step-by-Step + Quiz"')
  .where(p => p.file.ext === "md")
  .sort(p => p.Date ?? "9999-12-31", 'asc')
  .map(p => [
    p.file.link,
    p.Date ?? "—",
    p["HW #"] ?? "—",
    p.Studied ?? false,
    p["Quiz taken"] ?? false,
    p["Quiz score"] ?? "—",
    p["Day type"] ?? "—",
    p["Piece count"] ?? "—"
  ]);

dv.table(["Lesson", "Date", "HW #", "Studied", "Quiz taken", "Quiz score", "Day type", "Pieces"], rows);
```

## 📝 All DL Homework

```dataviewjs
const rows = dv.pages('"01 Learning/DL Homework Practice — MIT 6.7960"')
  .where(p => p.file.ext === "md")
  .sort(p => p["HW #"] ?? 999, 'asc')
  .map(p => [
    p.file.link,
    p["HW #"] ?? "—",
    p.Status ?? "—",
    p["Due Date"] ?? "—",
    p.Window ?? "—",
    p["Notes ready"] ?? false,
    p.PDF ?? "—"
  ]);

dv.table(["Homework", "HW #", "Status", "Due", "Window", "Notes ready", "PDF"], rows);
```
