---
cssclasses:
  - dashboard
obsidianUIMode: preview
banner: "[[99 Assets/Media/grpo_visual.png]]"
---
# 🏠 Knowledge Base Home

> [!tip] Today’s workspace
> Start here. This page is your Notion-like command center: open a collection, see what is active, and jump back into recently edited notes.

## 🚀 Launchpad

- 🧭 **[[00 Home/Vault Map|Vault Map]]** — how the whole system is organized
- 📥 **[[00 Home/Inbox/Inbox|Inbox]]** — default landing spot for new notes
- 📅 **[[00 Home/Daily Notes/2026-07-11|Today’s daily note]]**
- 🎓 **[[01 Learning/Learning Hub.base|Learning Hub]]** — courses, lessons, and homework
- 🔬 **[[02 Research/Research Hub.base|Research Hub]]** — thesis and paper reading
- 🧩 **[[03 Algorithms/Algorithms Hub.base|Algorithms Hub]]** — DSA practice
- 📚 **[[04 Reference/Reference Hub.base|Reference Hub]]** — deep technical notes
- 🖼️ **[[99 Assets/Assets Index|Assets Index]]** — media, canvases, and base files
- 🎨 **[[00 Home/Tag Color Map|Tag Color Map]]** — current tag palette

## 📊 Reading Status at a Glance

```dataviewjs
// Robust property getter for fields with spaces/case differences.
const prop = (page, ...names) => {
  for (const name of names) {
    if (page[name] !== undefined) return page[name];
    const key = name.toLowerCase().replaceAll(" ", "-");
    if (page[key] !== undefined) return page[key];
  }
  return undefined;
};

const collections = [
  {
    label: "🗂️ Document Hub",
    folder: "04 Reference/Document Hub",
    done: p => prop(p, "Status") === "Done",
    active: p => prop(p, "Status") === "In progress",
    todo: p => prop(p, "Status") === "Not started"
  },
  {
    label: "📖 Thesis Reading List",
    folder: "02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty",
    done: p => prop(p, "Reading Status") === "Read",
    active: p => prop(p, "Reading Status") === "Reading",
    todo: p => prop(p, "Reading Status") === "To Read"
  },
  {
    label: "👁️ CV Study Tracker",
    folder: "01 Learning/Computer Vision — Foundations Study Tracker",
    done: p => prop(p, "Status") === "Done",
    active: p => prop(p, "Status") === "In progress",
    todo: p => prop(p, "Status") === "Not started"
  },
  {
    label: "📝 DL Homework",
    folder: "01 Learning/DL Homework Practice — MIT 6.7960",
    done: p => prop(p, "Status") === "Done",
    active: p => prop(p, "Status") === "In progress",
    todo: p => prop(p, "Status") === "Not started"
  },
  {
    label: "🧠 DL Daily Lessons",
    folder: "01 Learning/DL Daily Lessons — Step-by-Step + Quiz",
    done: p => prop(p, "Studied") === true && prop(p, "Quiz taken") === true,
    active: p => prop(p, "Studied") === true && prop(p, "Quiz taken") !== true,
    todo: p => prop(p, "Studied") !== true
  },
  {
    label: "🧮 DSA",
    folder: "03 Algorithms/DSA",
    done: p => prop(p, "Status") === "Done",
    active: p => prop(p, "Status") === "To Review",
    todo: p => prop(p, "Status") === "To Solve"
  },
];

const rows = collections.map(c => {
  const pages = dv.pages(`"${c.folder}"`).where(p => p.file.ext === "md");
  const total = pages.length;
  const done = pages.where(c.done).length;
  const active = pages.where(c.active).length;
  const todo = pages.where(c.todo).length;
  const other = Math.max(total - done - active - todo, 0);
  const pct = total ? Math.round(done / total * 100) : 0;
  const bars = Math.round(pct / 10);
  const bar = "█".repeat(bars) + "░".repeat(10 - bars);
  return [c.label, total, `✅ ${done}`, `🟡 ${active}`, `⬜ ${todo}`, other ? `• ${other}` : "—", `\`${bar}\` ${pct}%`];
});

dv.table(["Base", "Total", "Done", "Active", "To Do", "Other", "Progress"], rows);
```

## 🎯 Focus Queue

> [!todo] Active learning / research
> These are pulled from your properties, so the queue updates automatically as you mark statuses.

```dataviewjs
const docs = dv.pages('"04 Reference/Document Hub"')
  .where(p => p.Status === "In progress")
  .sort(p => p.file.mtime, 'desc')
  .limit(6)
  .map(p => [p.file.link, p.Status ?? "—", p.Category ?? "—", dv.date(p.file.mtime).toFormat("MMM dd")]);

dv.table(["Active item", "Status", "Category", "Updated"], docs);
```

```dataviewjs
const reading = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"')
  .where(p => p["Reading Status"] === "Reading")
  .sort(p => `${p.Tier ?? ""}-${-(p.Year ?? 0)}`)
  .limit(8)
  .map(p => [p.file.link, p.Tier ?? "—", p.Topic ?? "—", p.Year ?? "—"]);

dv.table(["Reading now", "Tier", "Topic", "Year"], reading);
```

```dataviewjs
const practice = dv.pages('"03 Algorithms/DSA"')
  .where(p => p.Status === "To Solve" || p.Status === "To Review")
  .sort(p => `${p.Status ?? ""}-${p.Difficulty ?? ""}`, 'desc')
  .limit(8)
  .map(p => [p.file.link, p.Difficulty ?? "—", p.Topic ?? "—", p.Status ?? "—"]);

dv.table(["Practice / review", "Difficulty", "Topic", "Status"], practice);
```

## 🗂️ Full Inventories

- 🎓 [[01 Learning/Learning Hub.base|All learning documents and progress]]
- 🔬 [[02 Research/Research Hub.base|All research papers and thesis documents]]
- 🧩 [[03 Algorithms/Algorithms Hub.base|All DSA documents and practice status]]
- 📚 [[04 Reference/Reference Hub.base|All reference documents and status]]

## 🧱 Databases / Bases

- 🗂️ **[[04 Reference/Document Hub/Document Hub.base|Document Hub Base]]**
- 🧮 **[[03 Algorithms/DSA/DSA.base|DSA Base]]**
- 👁️ **[[01 Learning/Computer Vision — Foundations Study Tracker/Computer Vision — Foundations Study Tracker.base|CV Study Tracker Base]]**
- 🧠 **[[01 Learning/DL Daily Lessons — Step-by-Step + Quiz/DL Daily Lessons — Step-by-Step + Quiz.base|DL Daily Lessons Base]]**
- 📝 **[[01 Learning/DL Homework Practice — MIT 6.7960/DL Homework Practice — MIT 6.7960.base|DL Homework Base]]**
- 📖 **[[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base|Thesis Reading List Base]]**
- 🎓 **[[02 Research/Research Thesis|Research Thesis]]**

## ✏️ Recently Edited

```dataview
TABLE WITHOUT ID
  file.link AS "Note",
  dateformat(file.mtime, "MMM dd, HH:mm") AS "Modified",
  regexreplace(file.folder, "^.*/", "") AS "Folder"
FROM ""
WHERE !contains(file.path, "99 Assets") AND file.name != "Dashboard"
SORT file.mtime DESC
LIMIT 12
```

## 🔥 Git Activity

```dataviewjs
try {
  const { execSync } = require("child_process");
  const basePath = app.vault.adapter.basePath;
  const out = execSync("git log -8 --pretty=format:%h%x09%ar%x09%s", { cwd: basePath }).toString().trim();
  if (!out) dv.paragraph("No commits yet.");
  else dv.table(["Commit", "When", "Message"], out.split("\n").map(l => {
    const [hash, when, ...msg] = l.split("\t");
    return ["`" + hash + "`", when, msg.join("\t")];
  }));
} catch (e) {
  dv.paragraph("Git activity unavailable in this environment.");
}
```
