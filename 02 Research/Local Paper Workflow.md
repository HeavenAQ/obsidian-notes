---
base: "[[Document Hub.base]]"
Status: Done
Category:
  - Research
  - Reference Management
tags:
  - Zotero
  - PDF
  - Research-Workflow
Created time: 2026-07-12
Last updated time: 2026-07-12
---

# Local Paper Workflow

## Open the library

- Obsidian database: [[Local Paper Library.base]]
- CVPR roadmap: [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]
- Zotero collection: **CVPR 2027 — Temporal Self-Contact**
- Automatically maintained bibliography: `99 Assets/Bibliography/CVPR 2027 — Temporal Self-Contact.bib`

## Source of truth

| Information | Source of truth |
| --- | --- |
| PDF files | Local vault folder `99 Assets/Papers/CVPR 2027` |
| Bibliographic metadata and citation keys | Zotero + Better BibTeX |
| Reading status, priority, synthesis, and research links | Obsidian paper notes |
| PDF highlights | Zotero annotations imported through Zotero Integration, or direct PDF++ annotations |

## Daily workflow

1. Open a paper using its **Local PDF** property in [[Local Paper Library.base]].
2. Read and annotate in Zotero or PDF++.
3. If annotating in Zotero, run **Zotero Integration: Import CVPR paper** in Obsidian to pull annotations into a generated note.
4. Write conclusions in the existing named literature note rather than merely collecting highlights.
5. Change **Reading Status** to `Reading`, `Read`, or `Skim-Skip`; the paper bases and task board update automatically.
6. Cite using the **Citation Key** property. Better BibTeX automatically updates the project `.bib` file when Zotero metadata changes.

## Installed components

- Zotero Desktop 9
- Better BibTeX for Zotero
- Obsidian Zotero Integration
- PDF++

## Local-only storage rule

The downloaded PDF directory is excluded from Git to prevent approximately 180 MB of binary files from bloating repository history. The notes and bibliography remain version-controlled. Back up the PDF directory separately if needed.

The tracked download manifest is `99 Assets/Bibliography/CVPR 2027 — PDF Download Manifest.tsv`. If a PDF is deleted or corrupted, restore the complete local collection with:

```bash
python3 .obsidian/automation/download_cvpr_papers.py
```
