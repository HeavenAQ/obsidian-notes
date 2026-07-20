#!/usr/bin/env python3
"""Backfill the thesis paper library into Obsidian and Zotero.

The script is deliberately resumable and conservative:

1. It scans the existing thesis-reading Markdown notes.
2. It resolves and downloads only openly available PDFs.
3. It verifies the PDF signature before moving a file into the library.
4. It adds/updates ``Local PDF`` in the existing note without replacing its body.
5. With ``--sync-zotero``, it creates missing Zotero parents through Zotero's
   official Connector HTTP endpoint, uploads the PDF, moves the item to the
   configured collection, and writes Zotero/citation properties back to the note.

No Zotero SQLite database is ever modified directly. Existing Zotero matches and
existing valid local PDFs are reused, making repeat runs safe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import difflib
import html
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path


VAULT = Path(__file__).resolve().parents[2]
NOTES = VAULT / "02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty"
PDF_DIR = VAULT / "99 Assets/Papers/CVPR 2027"
MANIFEST = VAULT / "99 Assets/Bibliography/CVPR 2027 — PDF Download Manifest.tsv"
REPORT = VAULT / "99 Assets/Bibliography/CVPR 2027 — Paper Backfill Report.tsv"
ZOTERO_RPC = "http://127.0.0.1:23119/better-bibtex/json-rpc"
ZOTERO_CONNECTOR = "http://127.0.0.1:23119/connector"
ZOTERO_COLLECTION = "C1"  # CVPR 2027 — Temporal Self-Contact
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) paper-library-backfill/1.0"

# Verified author/preprint/repository copies for landing pages that either block
# automated retrieval or do not advertise their open PDF in standard metadata.
PDF_OVERRIDES = {
    "ARCTIC A Dataset for Dexterous Bimanual Hand-Object Manipulation":
        "https://arxiv.org/pdf/2204.13662",
    "Detecting Human-Object Contact in Images":
        "https://arxiv.org/pdf/2303.03373",
    "Rethinking the Evaluation Protocol of Domain Generalization":
        "https://arxiv.org/pdf/2305.15253",
    "Learning a Contact Potential Field for Modeling the Hand-Object Interaction":
        "https://arxiv.org/pdf/2012.00924",
    "Hierarchical Style-Aware Domain Generalization for Remote Physiological Measurement":
        "https://personal.hkust-gz.edu.cn/hedengbo/assets/publicationPDFs/Wang_IEEE_JBHI_2024a.pdf",
    "D-Touch Recognizing and Predicting Fine-grained Hand-face Touching Activities Using a Neck-mounted Wearable":
        "https://czhang.org/assets/pdf/DTouch.pdf",
    "Automatic Analysis of Naturalistic Hand-Over-Face Gestures":
        "https://api.repository.cam.ac.uk/server/api/core/bitstreams/30984f0b-b12b-4c8a-a544-c657a389fc83/content",
    "The CANDOR Corpus Insights from a Large Multimodal Dataset of Naturalistic Conversation":
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/ca/bb/sciadv.adf3197.PMC10065445.pdf",
    "FluencyBank Timestamped An Updated Data Set for Disfluency Detection and Automatic Intended Speech Recognition":
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/d0/6e/JSLHR-67-4203.PMC12379651.pdf",
    "ADABase A Multimodal Dataset for Cognitive Load Estimation":
        "https://mdpi-res.com/d_attachment/sensors/sensors-23-00340/article_deploy/sensors-23-00340-v2.pdf",
    "The Dual Functions of Adaptors":
        "https://mdpi-res.com/d_attachment/languages/languages-10-00231/article_deploy/languages-10-00231.pdf",
    "The Role of Self-Adaptors in Lexical Retrieval":
        "https://mdpi-res.com/d_attachment/languages/languages-10-00209/article_deploy/languages-10-00209.pdf",
}


@dataclass
class Paper:
    note: Path
    title: str
    url: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    doi: str = ""
    pdf: Path | None = None
    source: str = ""
    status: str = "PENDING"
    detail: str = ""


def request(url: str, *, data: bytes | None = None, headers=None, timeout=30):
    hdr = {"User-Agent": UA, "Accept": "text/html,application/json,application/pdf;q=0.9,*/*;q=0.5"}
    if headers:
        hdr.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=hdr), timeout=timeout)


def scalar(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.*?)\s*$", text, re.M)
    if not m:
        return ""
    return m.group(1).strip().strip('"\'')


def list_property(text: str, key: str) -> list[str]:
    m = re.search(rf"^{re.escape(key)}:\s*(.*?)$", text, re.M)
    if not m:
        return []
    inline = m.group(1).strip()
    if inline.startswith("["):
        return [x.strip().strip('"\'') for x in inline[1:-1].split(",") if x.strip()]
    start = m.end()
    values = []
    for line in text[start:].splitlines():
        if re.match(r"^\s+-\s+", line):
            values.append(re.sub(r"^\s+-\s+", "", line).strip().strip('"\''))
        elif line.strip() and not line[0].isspace():
            break
    return values


def load_papers() -> list[Paper]:
    papers = []
    for note in sorted(NOTES.glob("*.md")):
        text = note.read_text(encoding="utf-8")
        url = scalar(text, "Paper Link")
        if not url:
            continue
        doi = scalar(text, "DOI") or extract_doi(url)
        local = scalar(text, "Local PDF")
        pdf = None
        if local:
            rel = local.removeprefix("[[").removesuffix("]]" )
            pdf = VAULT / rel
        papers.append(Paper(
            note=note,
            title=note.stem,
            url=url,
            authors=scalar(text, "Authors"),
            year=scalar(text, "Year"),
            venue=scalar(text, "Venue"),
            topic=scalar(text, "Topic"),
            tags=list_property(text, "tags"),
            doi=doi,
            pdf=pdf,
        ))
    return papers


def extract_doi(value: str) -> str:
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", urllib.parse.unquote(value), re.I)
    return m.group(0).rstrip(".);,") if m else ""


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def valid_pdf(path: Path) -> bool:
    try:
        return path.stat().st_size > 10_000 and path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def absolute(base: str, candidate: str) -> str:
    return html.unescape(urllib.parse.urljoin(base, candidate.replace("\\/", "/")))


def page_candidates(url: str) -> tuple[list[str], str]:
    """Extract publisher-provided PDF metadata/links from a landing page."""
    try:
        with request(url, timeout=35) as response:
            content_type = response.headers.get_content_type()
            final_url = response.geturl()
            raw = response.read(3_000_000)
    except Exception:
        return [], ""
    if content_type == "application/pdf" or raw.startswith(b"%PDF-"):
        return [final_url], ""
    text = raw.decode("utf-8", "ignore")
    doi = extract_doi(text)
    candidates = []
    patterns = [
        r'<meta[^>]+(?:name|property)=["\'](?:citation_pdf_url|wkhealth_pdf_url|pdf_url)["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'](?:citation_pdf_url|wkhealth_pdf_url|pdf_url)["\']',
        r'<a[^>]+href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        r'"(?:pdfUrl|pdf_url|downloadUrl)"\s*:\s*"([^"]+)",?',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, re.I):
            candidate = absolute(final_url, match)
            if candidate.startswith(("http://", "https://")) and candidate not in candidates:
                candidates.append(candidate)
    return candidates[:12], doi


def openalex_candidates(paper: Paper) -> tuple[list[str], str]:
    """Use OpenAlex only as an open-access URL resolver for difficult legacy links."""
    if paper.doi:
        endpoint = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(paper.doi, safe="/")
    else:
        q = urllib.parse.urlencode({"search": paper.title, "per-page": 5})
        endpoint = "https://api.openalex.org/works?" + q
    try:
        with request(endpoint, headers={"Accept": "application/json"}, timeout=30) as r:
            payload = json.load(r)
    except Exception:
        return [], ""
    works = payload.get("results", []) if isinstance(payload, dict) and "results" in payload else [payload]
    best = None
    best_score = 0.0
    for work in works:
        score = title_similarity(paper.title, work.get("display_name", ""))
        if score > best_score:
            best, best_score = work, score
    if not best or best_score < 0.72:
        return [], ""
    urls = []
    locations = [best.get("best_oa_location") or {}] + (best.get("locations") or [])
    for loc in locations:
        u = loc.get("pdf_url") if isinstance(loc, dict) else None
        if u and u not in urls:
            urls.append(u)
    return urls, extract_doi(best.get("doi") or "")


def deterministic_candidates(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    out = []
    if host.endswith("arxiv.org"):
        ident = re.sub(r"^/(?:abs|pdf)/", "", path).removesuffix(".pdf").strip("/")
        if ident:
            out.append(f"https://arxiv.org/pdf/{ident}")
    elif host == "aclanthology.org":
        out.append(url.rstrip("/") + ".pdf")
    elif host == "www.isca-archive.org" and path.endswith(".html"):
        out.append(url[:-5] + ".pdf")
    elif host == "openaccess.thecvf.com" and "/html/" in path:
        out.append(url.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf"))
    elif host == "proceedings.neurips.cc" and "Abstract-Conference.html" in path:
        out.append(url.replace("Abstract-Conference.html", "Paper-Conference.pdf"))
    elif host == "www.mdpi.com":
        out.append(url.rstrip("/") + "/pdf")
    if path.lower().endswith(".pdf"):
        out.insert(0, url)
    return list(dict.fromkeys(out))


def download_candidate(url: str, target: Path) -> tuple[bool, str]:
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    result = subprocess.run([
        "curl", "-fL", "--retry", "3", "--retry-delay", "2",
        "--connect-timeout", "20", "--max-time", "300",
        "-A", UA, "-o", str(part), url,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and valid_pdf(part):
        part.replace(target)
        return True, ""
    detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "response was not a PDF"
    part.unlink(missing_ok=True)
    return False, detail


def resolve_and_download(paper: Paper, use_openalex=True) -> Paper:
    target = PDF_DIR / f"{paper.title}.pdf"
    paper.pdf = target
    if valid_pdf(target):
        paper.status, paper.source = "OK", paper.url
        return paper

    candidates = []
    if paper.title in PDF_OVERRIDES:
        candidates.append(PDF_OVERRIDES[paper.title])
    candidates.extend(deterministic_candidates(paper.url))
    page_urls, page_doi = page_candidates(paper.url)
    if page_doi and not paper.doi:
        paper.doi = page_doi
    candidates.extend(page_urls)
    if use_openalex and not any("arxiv.org/pdf/" in x for x in candidates):
        oa_urls, oa_doi = openalex_candidates(paper)
        candidates.extend(oa_urls)
        if oa_doi and not paper.doi:
            paper.doi = oa_doi

    seen = set()
    errors = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ok, detail = download_candidate(candidate, target)
        if ok:
            paper.status, paper.source = "OK", candidate
            return paper
        errors.append(f"{candidate}: {detail}")
    paper.status = "UNRESOLVED"
    paper.detail = " | ".join(errors[-3:]) if errors else "No open PDF candidate found"
    return paper


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def set_frontmatter_properties(note: Path, properties: dict[str, str]):
    text = note.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"No frontmatter: {note}")
    end = text.find("\n---", 4)
    if end < 0:
        raise ValueError(f"Unclosed frontmatter: {note}")
    fm, body = text[:end], text[end:]
    for key, value in properties.items():
        line = f"{key}: {yaml_quote(value)}"
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.M)
        if pattern.search(fm):
            fm = pattern.sub(line, fm, count=1)
        else:
            fm += "\n" + line
    updated = fm + body
    if updated != text:
        note.write_text(updated, encoding="utf-8")


def write_tracking(papers: list[Paper]):
    old = {}
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                old[row["note"]] = row
    for p in papers:
        if p.status in ("OK", "ZOTERO_OK") and p.pdf:
            previous = old.get(str(p.note.relative_to(VAULT)), {})
            old[str(p.note.relative_to(VAULT))] = {
                "note": str(p.note.relative_to(VAULT)),
                "pdf": str(p.pdf.relative_to(VAULT)),
                "source": p.source or previous.get("source") or p.url,
                "status": "OK",
            }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        fields = ["note", "pdf", "source", "status"]
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(old.values(), key=lambda r: r["note"]))

    with REPORT.open("w", encoding="utf-8", newline="") as f:
        fields = ["status", "note", "paper_link", "resolved_pdf", "detail"]
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        for p in sorted(papers, key=lambda p: (p.status, p.title)):
            writer.writerow({
                "status": p.status,
                "note": str(p.note.relative_to(VAULT)),
                "paper_link": p.url,
                "resolved_pdf": p.source,
                "detail": p.detail,
            })


def rpc(method: str, params: list):
    payload = json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}).encode()
    # Better BibTeX intentionally rejects browser-like cross-origin requests;
    # do not add the web User-Agent/Accept headers used by the PDF resolver.
    req = urllib.request.Request(
        ZOTERO_RPC, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.load(r)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("result")


def connector(endpoint: str, payload: dict):
    data = json.dumps(payload).encode()
    with request(
        f"{ZOTERO_CONNECTOR}/{endpoint}", data=data,
        headers={"Content-Type": "application/json", "X-Zotero-Connector-API-Version": "3"},
        timeout=120,
    ) as r:
        return r.status, r.read()


def zotero_matches(title: str):
    # Use Zotero's structured title search. A plain multi-word quick-search can
    # interpret terms too narrowly and miss an otherwise exact title.
    results = rpc("item.search", [[["title", "contains", title]]]) or []
    if not results:
        words = [x for x in re.findall(r"[A-Za-z0-9]+", title) if len(x) > 4]
        query = words[-1] if words else title
        results = rpc("item.search", [query]) or []
    return sorted(results, key=lambda x: title_similarity(title, x.get("title", "")), reverse=True)


def parse_authors(authors: str) -> list[dict]:
    creators = []
    for author in [x.strip() for x in authors.split(",") if x.strip()]:
        parts = author.split()
        if len(parts) == 1:
            creators.append({"lastName": parts[0], "creatorType": "author"})
        else:
            creators.append({"firstName": " ".join(parts[:-1]), "lastName": parts[-1], "creatorType": "author"})
    return creators


def zotero_item_type(venue: str) -> str:
    venue_u = venue.upper()
    conference_markers = ("CVPR", "ICCV", "ECCV", "WACV", "ACL", "EMNLP", "NAACL", "CHI", "ACM MM", "INTERSPEECH", "NEURIPS")
    if any(x in venue_u for x in conference_markers):
        return "conferencePaper"
    if venue_u == "ARXIV" or not venue:
        return "preprint"
    return "journalArticle"


def create_zotero_parent(paper: Paper):
    session = "backfill-" + uuid.uuid4().hex
    connector_id = "paper-1"
    item = {
        "id": connector_id,
        "itemType": zotero_item_type(paper.venue),
        "title": paper.title,
        "creators": parse_authors(paper.authors),
        "date": paper.year,
        "url": paper.url,
        "DOI": paper.doi,
        "publicationTitle": paper.venue if paper.venue.lower() != "arxiv" else "",
        "repository": "arXiv" if paper.venue.lower() == "arxiv" else "",
        "tags": [{"tag": t} for t in paper.tags + ([paper.topic] if paper.topic else [])],
    }
    # Remove fields that are empty or invalid for some Zotero item types.
    item = {k: v for k, v in item.items() if v not in ("", [], None)}
    if item["itemType"] != "preprint":
        item.pop("repository", None)
    status, _ = connector("saveItems", {"sessionID": session, "uri": paper.url, "items": [item]})
    if status != 201:
        raise RuntimeError(f"saveItems returned {status}")
    connector("updateSession", {"sessionID": session, "target": ZOTERO_COLLECTION, "tags": "", "note": ""})

    # Connector file upload creates a Zotero-owned attachment. The verified vault
    # copy remains available to Obsidian and as an offline repair source.
    metadata = json.dumps({
        "sessionID": session,
        "parentItemID": connector_id,
        "title": "Full Text PDF",
        "url": paper.source or paper.url,
    }, ensure_ascii=False)
    pdf_data = paper.pdf.read_bytes()
    req = urllib.request.Request(
        f"{ZOTERO_CONNECTOR}/saveAttachment",
        data=pdf_data,
        headers={
            "User-Agent": UA,
            "Content-Type": "application/pdf",
            "Content-Length": str(len(pdf_data)),
            "X-Metadata": metadata,
            "X-Zotero-Connector-API-Version": "3",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 201:
            raise RuntimeError(f"saveAttachment returned {r.status}")


def sync_one_zotero(paper: Paper) -> Paper:
    if paper.status != "OK" or not paper.pdf or not valid_pdf(paper.pdf):
        return paper
    try:
        matches = zotero_matches(paper.title)
        match = matches[0] if matches and title_similarity(paper.title, matches[0].get("title", "")) >= 0.90 else None
        if not match:
            create_zotero_parent(paper)
            # Better BibTeX assigns its key asynchronously.
            for _ in range(12):
                time.sleep(0.5)
                matches = zotero_matches(paper.title)
                if matches and title_similarity(paper.title, matches[0].get("title", "")) >= 0.90:
                    match = matches[0]
                    break
        if not match:
            raise RuntimeError("created item could not be found through Better BibTeX")

        zotero_key = match["id"].rstrip("/").split("/")[-1]
        citekey = match.get("citation-key") or match.get("citekey") or ""
        attachments = rpc("item.attachments", [citekey]) if citekey else []
        open_uri = ""
        for attachment in attachments or []:
            if attachment.get("open", "").startswith("zotero://open-pdf/"):
                open_uri = attachment["open"]
                break
        props = {
            "Local PDF": f"[[{paper.pdf.relative_to(VAULT)}]]",
            "Zotero URI": f"zotero://select/library/items/{zotero_key}",
        }
        if citekey:
            props["Citation Key"] = citekey
        if open_uri:
            props["Zotero PDF URI"] = open_uri
        set_frontmatter_properties(paper.note, props)
        paper.status = "ZOTERO_OK"
    except Exception as e:
        paper.status = "ZOTERO_FAILED"
        paper.detail = str(e)
    return paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-zotero", action="store_true", help="create/match Zotero items and attach PDFs")
    parser.add_argument("--workers", type=int, default=4, help="parallel PDF resolver/download workers")
    parser.add_argument("--limit", type=int, help="process at most this many missing papers")
    parser.add_argument("--no-openalex", action="store_true", help="disable OpenAlex OA fallback")
    parser.add_argument("--all", action="store_true", help="also re-audit notes that already have Local PDF")
    parser.add_argument("--metadata-only", action="store_true", help="skip downloads and refresh Zotero/note metadata for valid local PDFs")
    args = parser.parse_args()

    papers = load_papers()
    if args.metadata_only:
        pending = [p for p in papers if p.pdf and valid_pdf(p.pdf)]
    else:
        pending = papers if args.all else [p for p in papers if not (p.pdf and valid_pdf(p.pdf))]
    if args.limit:
        pending = pending[:args.limit]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    action = "existing PDFs for metadata refresh" if args.metadata_only else "missing PDFs"
    print(f"Scanning {len(papers)} paper notes; processing {len(pending)} {action}")

    completed = []
    if args.metadata_only:
        for paper in pending:
            # Do not replace a previously resolved direct-PDF manifest source
            # with the paper's landing page during a metadata refresh.
            paper.status, paper.source = "OK", ""
            completed.append(paper)
        print(f"Using {len(completed)} already verified local PDFs")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(resolve_and_download, p, not args.no_openalex): p for p in pending}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                paper = future.result()
                completed.append(paper)
                print(f"[{i:03}/{len(pending):03}] {paper.status:10} {paper.title}")
                if paper.status == "OK" and paper.pdf:
                    set_frontmatter_properties(paper.note, {
                        "Local PDF": f"[[{paper.pdf.relative_to(VAULT)}]]",
                    })

    write_tracking(completed)

    if args.sync_zotero:
        try:
            ready = rpc("api.ready", [])
            print(f"Zotero ready: {ready}")
        except Exception as e:
            print(f"Zotero/Better BibTeX unavailable: {e}", file=sys.stderr)
            return 2
        for i, paper in enumerate([p for p in completed if p.status == "OK"], 1):
            paper = sync_one_zotero(paper)
            print(f"[Z {i:03}] {paper.status:13} {paper.title}")
        # Write a complete audit, not only the subset touched by this run.
        processed = {p.note: p for p in completed}
        full_report = list(completed)
        for paper in papers:
            if paper.note in processed:
                continue
            text = paper.note.read_text(encoding="utf-8", errors="ignore")
            if paper.pdf and valid_pdf(paper.pdf):
                paper.status = "ZOTERO_OK" if scalar(text, "Zotero URI") else "OK"
            else:
                paper.status = "UNRESOLVED"
                paper.detail = "No verified open PDF is currently available in the local library"
            full_report.append(paper)
        write_tracking(full_report)

    counts = {}
    for p in completed:
        counts[p.status] = counts.get(p.status, 0) + 1
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("Report:", REPORT.relative_to(VAULT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
