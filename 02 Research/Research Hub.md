---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🔬 Research Hub

## Research spaces

- 🎓 **[[02 Research/Research Thesis|Research Thesis]]**
- 📖 **[[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base|Thesis Reading List Base]]**

## Reading pipeline

```dataviewjs
const now = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"')
  .where(p => p["Reading Status"] === "Reading")
  .sort(p => `${p.Tier ?? ""}-${-(p.Year ?? 0)}`)
  .map(p => [p.file.link, p.Tier ?? "—", p.Topic ?? "—", p.Year ?? "—", p.Venue ?? "—"]);

dv.table(["Reading now", "Tier", "Topic", "Year", "Venue"], now);
```

```dataviewjs
const essential = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"')
  .where(p => p.Tier === "Essential" && p["Reading Status"] !== "Read")
  .sort(p => p.Year ?? 0, 'desc')
  .limit(12)
  .map(p => [p.file.link, p.Topic ?? "—", p.Year ?? "—", p.Venue ?? "—", p["Reading Status"] ?? "—"]);

dv.table(["Essential to read", "Topic", "Year", "Venue", "Reading Status"], essential);
```

```dataviewjs
const recent = dv.pages('"02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"')
  .sort(p => p.file.mtime, 'desc')
  .limit(10)
  .map(p => [p.file.link, p.Topic ?? "—", p["Reading Status"] ?? "—", dv.date(p.file.mtime).toFormat("MMM dd")]);

dv.table(["Recently updated paper", "Topic", "Reading Status", "Updated"], recent);
```
