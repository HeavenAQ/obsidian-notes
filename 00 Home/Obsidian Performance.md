---
base: "[[Document Hub.base]]"
Status: Done
Category:
  - Obsidian
  - Maintenance
tags:
  - Performance
  - Obsidian
Created time: 2026-07-13
Last updated time: 2026-07-13
---

# Obsidian Performance

## 2026-07-13 optimization

The renderer was observed consuming approximately **99% CPU continuously**. The primary problem was the custom property-tag color plugin: every DOM/class mutation triggered a full-document scan, thousands of inline style writes, and interaction with a 620 KB tag stylesheet.

Changes applied:

- Replaced full-document mutation rescans with incremental processing of newly inserted property/Base pills.
- Removed descendant-by-descendant inline styling and retained consistent Notion colors through two CSS variables.
- Reduced `tag-colors.css` from approximately **620 KB / 12,293 lines** to **1.7 KB / 59 lines**.
- Disabled make.md while preserving its settings; native File Explorer, Bases, Iconic, and the dashboard provide the active organization layer.
- Disabled Importer because migration is complete.
- Disabled Git backup on every file modification and reduced source-control refresh frequency. Manual and 30-minute Git backups remain available.
- Reduced task-board automation from every 30 minutes to every 2 hours, while retaining event-driven refresh after a 90-second quiet period and the manual refresh command.
- Removed hidden Base tabs and inactive sidebar tools from the saved workspace. Files and plugins were not deleted.
- Removed the synchronous `git log` shell command from [[Dashboard]].

The saved workspace was reduced from **24 loaded leaves** to **12**. After restart, renderer samples fell from a sustained ~99% to a variable range generally below 40% while this chat/tool output was actively updating.

## Rollback

The pre-change plugin, CSS, Git settings, community plugin list, and workspace are stored in `.obsidian/performance-backup-2026-07-13/`.

make.md and Importer remain installed but disabled, so they can be re-enabled without reinstalling.

