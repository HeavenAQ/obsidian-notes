#!/usr/bin/env python3
"""Generate Deadline Triage and Task Kanban deadline lanes.

Rules:
- Include explicit Due Date pages.
- Include homework sub-deadlines inferred from dated checklist lines and homework Window Day N plans.
- Include CV study Date pages and DL daily lesson Date pages when unfinished.
- Exclude research Assigned Date backlog from urgent deadline lanes by design.
"""
import argparse
import datetime as dt
import re
from pathlib import Path
from collections import Counter

MONTHS = {m: i for i, m in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], 1)}
MONTH_RE = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
ROOT = Path.cwd()
TRIAGE = ROOT / '00 Home/Tasks/Deadline Triage.md'
KANBAN = ROOT / '00 Home/Tasks/Task Kanban Board.md'


def parse_fm(text):
    if not text.startswith('---'):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            end = i
            break
    if end is None:
        return {}, text
    data = {}
    cur = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line.startswith((' ', '\t', '-')):
            m = re.match(r'^([^:\n]+):\s*(.*)$', line)
            if m:
                cur = m.group(1).strip().strip('"\'')
                val = m.group(2).strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                data[cur] = [] if val == '' else val
                continue
        if cur and re.match(r'^\s*-\s+', line):
            item = re.sub(r'^\s*-\s+', '', line).strip().strip('"')
            if not isinstance(data.get(cur), list):
                data[cur] = [] if data.get(cur) in ('', None) else [data[cur]]
            data[cur].append(item)
    return data, '\n'.join(lines[end+1:])


def parse_date(value):
    if not value:
        return None
    s = str(value).strip().strip('"')
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def date_from_month_day(mon, day, year):
    return dt.date(year, MONTHS[mon[:3].lower()], int(day))


def listv(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    return [str(value)]


def has_val(value, target):
    return target in listv(value) or str(value) == target


def is_page_done(fm):
    if has_val(fm.get('Status'), 'Done'):
        return True
    if has_val(fm.get('Reading Status'), 'Read'):
        return True
    if str(fm.get('Studied')).lower() == 'true' and str(fm.get('Quiz taken')).lower() == 'true':
        return True
    return False


def wikilink(path):
    path = Path(path)
    try:
        path = path.relative_to(ROOT)
    except ValueError:
        pass
    text = str(path)
    no_ext = text[:-3] if text.endswith('.md') else text
    label = Path(no_ext).name
    return f'[[{no_ext}|{label}]]'


def clean_md(text, maxlen=190):
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[*_`#>]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return (text[:maxlen-1] + '…') if len(text) > maxlen else text


def add(items, today, group, due, text, source, tag, reason):
    if not due:
        return
    status = 'overdue' if due < today else 'today' if due == today else 'upcoming'
    items.append({
        'group': group,
        'due': due,
        'text': text,
        'source': source,
        'tag': tag,
        'reason': reason,
        'status': status,
    })


def collect_items(today):
    items = []
    # Explicit due dates.
    for md in sorted(ROOT.rglob('*.md')):
        if '.obsidian' in md.parts or md == TRIAGE:
            continue
        fm, body = parse_fm(md.read_text(errors='ignore'))
        due = parse_date(fm.get('Due Date'))
        if due and not is_page_done(fm):
            md_s = str(md)
            area = 'DSA' if '03 Algorithms/DSA' in md_s else 'Homework' if 'DL Homework Practice' in md_s else 'Explicit due'
            tag = '#deadline-dsa' if area == 'DSA' else '#deadline-homework' if area == 'Homework' else '#deadline-explicit'
            add(items, today, f'Explicit due dates — {area}', due, f'{wikilink(md)} — page due date', md, tag, 'Due Date property')

    # CV study dates.
    cv_dir = ROOT / '01.1 Computer Vision — Foundations Study Tracker'
    if cv_dir.exists():
        for md in sorted(cv_dir.glob('*.md')):
            fm, body = parse_fm(md.read_text(errors='ignore'))
            due = parse_date(fm.get('Date'))
            if due and due <= today and not is_page_done(fm):
                status = ', '.join(listv(fm.get('Status'))) or 'open'
                add(items, today, 'Implicit dated study pages — CV', due, f'{wikilink(md)} — scheduled CV study page ({status})', md, '#deadline-cv', 'Date property treated as scheduled study deadline')

    # DL daily lessons.
    dl_dir = ROOT / '01.2 DL Daily Lessons — Step-by-Step + Quiz'
    if dl_dir.exists():
        for md in sorted(dl_dir.glob('*.md')):
            fm, body = parse_fm(md.read_text(errors='ignore'))
            due = parse_date(fm.get('Date'))
            if due and due <= today and not is_page_done(fm):
                bits = []
                if str(fm.get('Studied')).lower() != 'true':
                    bits.append('study not marked done')
                if str(fm.get('Quiz taken')).lower() != 'true':
                    bits.append('quiz not taken')
                add(items, today, 'Implicit dated lessons — DL daily', due, f'{wikilink(md)} — {", ".join(bits)}', md, '#deadline-dl-daily', 'Date property + Studied/Quiz unchecked')

    # Homework sub-deadlines.
    hw_dir = ROOT / '01.3 DL Homework Practice — MIT 6.7960'
    if hw_dir.exists():
        for md in sorted(hw_dir.glob('Homework*.md')):
            fm, body = parse_fm(md.read_text(errors='ignore'))
            due_page = parse_date(fm.get('Due Date'))
            window = str(fm.get('Window', ''))
            win_m = re.search(rf'({MONTH_RE})\s+(\d{{1,2}})\s*[–-]\s*({MONTH_RE})?\s*(\d{{1,2}}),\s*(\d{{4}})', window, re.I)
            win_start = None
            if win_m:
                mon1, day1, _mon2, _day2, yr = win_m.groups()
                win_start = date_from_month_day(mon1, day1, int(yr))
            for line in body.splitlines():
                if not re.match(r'\s*- \[ \]', line):
                    continue
                due = None
                reason = None
                m = re.search(rf'\*\*({MONTH_RE})\s+(\d{{1,2}})\*\*', line, re.I)
                if m:
                    due = date_from_month_day(m.group(1), m.group(2), due_page.year if due_page else today.year)
                    reason = 'dated homework checklist item'
                else:
                    m = re.search(r'\*\*Day\s+(\d+)\b', line, re.I)
                    if m and win_start:
                        due = win_start + dt.timedelta(days=int(m.group(1)) - 1)
                        reason = f'Day {m.group(1)} inferred from homework window'
                if due:
                    txt = clean_md(re.sub(r'^\s*- \[ \]\s*', '', line))
                    add(items, today, 'Implicit homework sub-deadlines', due, f'{wikilink(md)} — {txt}', md, '#deadline-homework-plan', reason)

    # Note: Research Assigned Date backlog is intentionally excluded from urgent/deadline lanes.
    # It belongs in the Research Hub / thesis reading base, not in daily overdue pressure.

    seen = set()
    unique = []
    for item in items:
        key = (item['due'], item['text'], str(item['source']))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return sorted(unique, key=lambda x: (x['due'], x['group'], x['text']))


def write_triage(items, today):
    counts = Counter(item['status'] for item in items)
    lines = []
    lines += ['---', 'cssclasses:', '  - dashboard', 'Status: In-Progress', 'Category:', '  - Task-Management', '---', '']
    lines += ['# 🧨 Deadline Triage', '', f'> Auto-generated on {today.isoformat()} by `.obsidian/automation/deadline_triage.py`.', '> Research assigned-reading backlog is intentionally excluded from urgent overdue/today deadline lanes.', '']
    lines += ['## Summary', '', f'- Total surfaced deadline/backlog items: **{len(items)}**', f'- Overdue: **{counts["overdue"]}**', f'- Due today: **{counts["today"]}**', f'- Upcoming: **{counts["upcoming"]}**', '']
    lines += ['## How to use this list', '', '- Work top to bottom inside **Overdue** first.', '- When the real source task is completed, check the generated triage checkbox here too.', '- Research reading remains visible in the Research Hub/Base instead of urgent deadline lanes.', '']
    labels = {'overdue': '🚨 Overdue / due now', 'today': '📅 Due today', 'upcoming': '🗓️ Upcoming'}
    for status in ['overdue', 'today', 'upcoming']:
        subset = [item for item in items if item['status'] == status]
        lines += [f'## {labels[status]} ({len(subset)})', '']
        current_group = None
        for item in subset:
            if item['group'] != current_group:
                current_group = item['group']
                lines += [f'### {current_group}', '']
            lines.append(f'- [ ] {item["text"]} 📅 {item["due"].isoformat()} {item["tag"]} <!-- inferred: {item["reason"]} -->')
        lines.append('')
    TRIAGE.parent.mkdir(parents=True, exist_ok=True)
    TRIAGE.write_text('\n'.join(lines), encoding='utf-8')


def write_kanban(items):
    by = {'overdue': [], 'today': [], 'upcoming': []}
    for item in items:
        content = f'{item["text"]} 📅 {item["due"].isoformat()} {item["tag"]}'
        by[item['status']].append((item['due'], content))
    for key in by:
        by[key].sort(key=lambda x: (x[0], x[1]))

    settings = '''%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[],"show-checkboxes":true,"new-note-folder":"00 Home/Tasks","date-format":"YYYY-MM-DD","date-display-format":"YYYY-MM-DD","link-date-to-daily-note":true,"tag-colors":[{"tagKey":"#task/research","color":"rgba(51, 126, 169, 0.16)"},{"tagKey":"#task/homework","color":"rgba(68, 131, 97, 0.16)"},{"tagKey":"#task/admin","color":"rgba(203, 145, 47, 0.16)"},{"tagKey":"#task/waiting","color":"rgba(212, 76, 71, 0.14)"},{"tagKey":"#deadline-homework-plan","color":"rgba(68, 131, 97, 0.16)"},{"tagKey":"#deadline-homework","color":"rgba(203, 145, 47, 0.16)"},{"tagKey":"#deadline-dsa","color":"rgba(144, 101, 176, 0.16)"},{"tagKey":"#deadline-cv","color":"rgba(68, 131, 97, 0.16)"},{"tagKey":"#deadline-dl-daily","color":"rgba(217, 115, 13, 0.16)"}]}
```
%%'''
    lines = ['---', 'kanban-plugin: board', 'cssclasses:', '  - dashboard', 'Status: In-Progress', 'Category:', '  - Task-Management', '---', '']
    lines += ['## 📥 Inbox', '', '- [[00 Home/Tasks/Task Inbox]] — Process loose captured tasks #task/inbox', '- [[00 Home/Tasks/Task Command Center]] — Triage tasks with no due/scheduled/start date #task/admin', '- [[00 Home/Daily Notes/2026-07-12]] — Review today’s Top 3 #task/admin', '']
    lines += ['## 🎯 Next', '', '- [[00 Home/Tasks/Deadline Triage]] — Work through surfaced overdue/deadline items #deadline-triage', '- [[02 Research/Research Thesis]] — Discuss two-layer plan and ML-heavy methods with Prof. Sekine #task/research', '- [[01.3 DL Homework Practice — MIT 6.7960/Homework 1]] — Continue HW1 prep and implementation plan #task/homework', '']
    lines += ['## 🚧 Doing', '', '- [[00 Home/Tasks/Task Command Center]] — Clear Overdue / Today queue #task/admin', '']
    lines += ['## 🧱 Waiting / Blocked', '', '- [[02 Research/Research Thesis]] — Secure dataset access and confirm ethics/usage approvals #task/waiting #task/research', '']
    lane_names = {'overdue': '🚨 Deadline Overdue', 'today': '📅 Deadline Today', 'upcoming': '🗓️ Deadline Upcoming'}
    for status in ['overdue', 'today', 'upcoming']:
        entries = by[status]
        lines += [f'## {lane_names[status]} ({len(entries)})', '']
        if entries:
            for _due, content in entries:
                lines.append(f'- {content}')
        else:
            lines.append('- Nothing here.')
        lines.append('')
    lines += ['## ✅ Done', '', '- [[00 Home/Tasks/Productivity Plugin Setup]] — Productivity/task stack installed #task/admin', '- [[00 Home/Tasks/Task Workflow]] — Task workflow documented #task/admin', '', settings, '']
    KANBAN.parent.mkdir(parents=True, exist_ok=True)
    KANBAN.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', help='Override today as YYYY-MM-DD')
    args = parser.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    items = collect_items(today)
    write_triage(items, today)
    write_kanban(items)
    counts = Counter(item['status'] for item in items)
    print(f'Generated {len(items)} items: overdue={counts["overdue"]}, today={counts["today"]}, upcoming={counts["upcoming"]}')


if __name__ == '__main__':
    main()
