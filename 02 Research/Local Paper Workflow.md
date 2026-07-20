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
Last updated time: 2026-07-20
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
| PDF library and annotation copy | Zotero collection **CVPR 2027 — Temporal Self-Contact** |
| Verified offline/Obsidian PDF mirror | Local vault folder `99 Assets/Papers/CVPR 2027` |
| Bibliographic metadata and citation keys | Zotero + Better BibTeX |
| Reading status, priority, synthesis, and research links | Obsidian paper notes |
| PDF highlights | Zotero annotations imported through Zotero Integration, or direct PDF++ annotations |

## Capture a new paper — recommended two-action workflow

1. Open the paper's **article/abstract page** in the browser rather than opening a bare PDF when possible.
2. Click **Save to Zotero** in the Zotero Connector and choose **CVPR 2027 — Temporal Self-Contact**. Zotero handles the site's metadata translator, authenticated access, duplicate detection, and available PDF.
3. In Obsidian, press **⌘⌥I** (`Zotero Integration: Import CVPR paper`) and select the new item.

The Obsidian import uses [[00 Home/Templates/Zotero Literature Note Template]] and writes imported notes/annotations under `02 Research/Zotero Imports`. Zotero remains responsible for the bibliographic item and its annotation copy; Obsidian remains responsible for synthesis, research links, status, and project-specific interpretation.

> [!tip] Why this is not a blind URL watcher
> The one Connector click preserves publisher login/cookies and lets Zotero's maintained translators decide what the page represents. A background downloader cannot reliably reproduce that behavior across paywalls and publisher anti-bot systems.

## Backfill and repair automation

The resumable backfill scans the 151 notes in the thesis reading list, resolves openly available PDFs, verifies the PDF signature, adds the items to the Zotero collection, and writes these properties back into each matching note:

- `Citation Key`
- `Local PDF`
- `Zotero URI`
- `Zotero PDF URI`

Run the complete backfill for newly added paper notes:

```bash
python3 .obsidian/automation/backfill_paper_library.py --workers 4 --sync-zotero
```

Refresh Zotero links and citation keys without accessing publishers again:

```bash
python3 .obsidian/automation/backfill_paper_library.py --metadata-only --sync-zotero
```

The process is safe to rerun: it reuses valid PDFs and title-matched Zotero items rather than creating them again. It never writes directly to Zotero's SQLite database. The detailed audit is tracked in `99 Assets/Bibliography/CVPR 2027 — Paper Backfill Report.tsv`.

### Backfill completed on 2026-07-20

- **122 / 151 papers** downloaded, verified, organized in Obsidian, and linked to Zotero.
- **29 / 151 papers** remain unresolved because no downloadable open copy was found or the publisher/repository blocked automated retrieval.
- Unresolved papers remain untouched in the reading list and are recorded in the report for manual Zotero Connector capture through institutional access.

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

The downloaded PDF directory is excluded from Git to prevent the binary library (currently approximately 929 MB) from bloating repository history. The notes, automation, manifest, and bibliography remain version-controlled. Back up the PDF directory separately if needed.

The tracked download manifest is `99 Assets/Bibliography/CVPR 2027 — PDF Download Manifest.tsv`. If a PDF is deleted or corrupted, restore the complete local collection with:

```bash
python3 .obsidian/automation/download_cvpr_papers.py
```
