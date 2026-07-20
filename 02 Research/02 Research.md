---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🔬 Research Hub

> [!important] Primary hub
> This hub is now maintained as a Base: [[Research Hub.base|Research Hub Base]].

## Research spaces

- 🎓 **[[Research Thesis|Research Thesis]]**
- 📖 **[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base|Thesis Reading List Base]]**
- 🏷️ **[[00 Home/Tag Normalization Report|Tag Normalization Report]]**

## 📊 Research Reading Progress

```dataviewjs
const pages = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"').where(p => p.file.ext === "md");
const statuses = ["Reading", "To-Read", "Read", "Skim-Skip"];
const rows = statuses.map(s => [s, pages.where(p => p["Reading Status"] === s).length]);
const total = pages.length;
const read = pages.where(p => p["Reading Status"] === "Read").length;
const pct = total ? Math.round(read / total * 100) : 0;
rows.push(["Total papers", total]);
rows.push(["Completion", `${read}/${total} (${pct}%)`]);
dv.table(["Status", "Count"], rows);
```

## 📚 All Thesis Reading List Papers

```dataviewjs
const tagList = p => {
  const tags = p.file.tags && p.file.tags.length ? p.file.tags : (p.tags ?? []);
  return Array.isArray(tags) ? tags.map(t => String(t).replace(/^#/, "")).join(", ") : String(tags ?? "—");
};

const rows = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"')
  .where(p => p.file.ext === "md")
  .sort(p => p["Assigned Date"] ?? "0000-00-00", 'desc')
  .sort(p => p.Year ?? 0, 'desc')
  .map(p => [
    p.file.link,
    p["Assigned Date"] ?? "—",
    p["Should Refer"] ?? "—",
    p.Relatedness ?? "—",
    p.snippet ?? "—",
    p["Reading Status"] ?? "—",
    tagList(p),
    p.Authors ?? "—",
    p["Code Link"] ?? "—",
    p["Paper Link"] ?? "—",
    p.Tier ?? "—",
    p.Topic ?? "—",
    p.Venue ?? "—",
    p.Year ?? "—",
    p.Dataset ?? "—"
  ]);

dv.table([
  "Title", "Assigned Date", "Should Refer", "Relatedness", "snippet", "Reading Status", "Tags", "Authors", "Code Link", "Paper Link", "Tier", "Topic", "Venue", "Year", "Dataset"
], rows);
```

## 🎓 Research Thesis / Other Research Docs

```dataviewjs
const rows = dv.pages('"02 Research"')
  .where(p => p.file.ext === "md" && !p.file.path.includes("Thesis Reading List — Self-Adaptors"))
  .sort(p => p.file.name, 'asc')
  .map(p => [p.file.link, p.file.folder, dv.date(p.file.mtime).toFormat("MMM dd, HH:mm")]);

dv.table(["Document", "Folder", "Updated"], rows);
```
