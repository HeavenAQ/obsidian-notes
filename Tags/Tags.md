---
cssclasses:
  - dashboard
obsidianUIMode: preview
---

# 🏷️ Tags

> [!info]
> Vault-wide tag index, sorted by usage.

```dataviewjs
const counts = new Map();

for (const page of dv.pages()) {
  for (const tag of (page.file.tags ?? [])) {
    counts.set(tag, (counts.get(tag) ?? 0) + 1);
  }
}

const rows = [...counts.entries()]
  .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  .map(([tag, count]) => [tag, count]);

dv.table(["Tag", "Notes"], rows);
```
