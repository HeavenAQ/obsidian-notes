---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🏠 Home

## 📚 Document Bases

- 🗂️ **[[Notion/Document Hub/Document Hub.base|Document Hub]]**
- 🧮 **[[Notion/DSA/DSA.base|DSA Practice]]**
- 👁️ **[[Notion/Computer Vision — Foundations Study Tracker/Computer Vision — Foundations Study Tracker.base|CV Study Tracker]]**
- 🧠 **[[Notion/DL Daily Lessons — Step-by-Step + Quiz/DL Daily Lessons — Step-by-Step + Quiz.base|DL Daily Lessons]]**
- 📝 **[[Notion/DL Homework Practice — MIT 6.7960/DL Homework Practice — MIT 6.7960.base|DL Homework · MIT 6.7960]]**
- 📖 **[[Notion/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base|Thesis Reading List]]**
- 🎓 **[[Notion/Research Thesis|Research Thesis]]**

## 🔥 Commit Activity

```dataviewjs
try {
    const { execSync } = require("child_process");
    const basePath = app.vault.adapter.basePath;
    const out = execSync("git log --pretty=format:%ad --date=short", { cwd: basePath }).toString().trim();

    const counts = {};
    if (out) for (const d of out.split("\n")) counts[d] = (counts[d] || 0) + 1;

    const calendarData = {
        year: new Date().getFullYear(),
        colors: { green: ["#c6e48b", "#7bc96f", "#49af5d", "#2e8840", "#196127"] },
        showCurrentDayBorder: true,
        entries: Object.entries(counts).map(([date, n]) => ({
            date, intensity: n, content: ""
        }))
    };
    renderHeatmapCalendar(this.container, calendarData);
} catch (e) {
    dv.paragraph("⚠️ Heatmap unavailable: " + e.message);
}
```

## 🕐 Recent Commits

```dataviewjs
try {
    const { execSync } = require("child_process");
    const basePath = app.vault.adapter.basePath;
    const out = execSync("git log -8 --pretty=format:%h%x09%ar%x09%s", { cwd: basePath }).toString().trim();

    if (!out) {
        dv.paragraph("No commits yet.");
    } else {
        const rows = out.split("\n").map(l => {
            const [hash, when, ...msg] = l.split("\t");
            return ["`" + hash + "`", when, msg.join("\t")];
        });
        dv.table(["Commit", "When", "Message"], rows);
    }
} catch (e) {
    dv.paragraph("⚠️ Git log unavailable: " + e.message);
}
```

## ✏️ Recently Edited

```dataview
TABLE WITHOUT ID
    file.link AS "Note",
    dateformat(file.mtime, "MMM dd, HH:mm") AS "Modified"
FROM ""
WHERE file.name != "Dashboard"
SORT file.mtime DESC
LIMIT 8
```
