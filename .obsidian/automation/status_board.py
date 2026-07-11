#!/usr/bin/env python3
"""Collect page status cards for the automated Kanban board.

This reads Obsidian/Notion-style frontmatter properties such as Status,
Reading Status, Studied, Quiz taken, Due Date, Category, Topic, and Tier.
"""
import datetime as dt
import re
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.cwd()
INCLUDE_ROOTS = ('01 Learning', '02 Research', '03 Algorithms', '04 Reference')
EXCLUDE_PARTS = {'.obsidian', '99 Assets', '90 Archive'}


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
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', str(value))
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def listv(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    return [str(value)]


def firstv(value, default=''):
    vals = listv(value)
    return vals[0] if vals else default


def has(value, target):
    return target in listv(value) or str(value) == target


def wikilink(path):
    path = Path(path)
    try:
        path = path.relative_to(ROOT)
    except ValueError:
        pass
    text = str(path)
    no_ext = text[:-3] if text.endswith('.md') else text
    return f'[[{no_ext}|{Path(no_ext).name}]]'


def should_include(md):
    if md.suffix != '.md':
        return False
    if any(part in EXCLUDE_PARTS for part in md.parts):
        return False
    rel = str(md.relative_to(ROOT)) if md.is_absolute() else str(md)
    return rel.startswith(INCLUDE_ROOTS)


def classify(fm):
    status_vals = set(listv(fm.get('Status')))
    reading_vals = set(listv(fm.get('Reading Status')))
    studied = str(fm.get('Studied')).lower() == 'true'
    quiz = str(fm.get('Quiz taken')).lower() == 'true'

    if 'Done' in status_vals or 'Read' in reading_vals or (studied and quiz):
        return 'done'
    if 'In-Progress' in status_vals or 'Reading' in reading_vals or studied:
        return 'doing'
    if 'To-Review' in status_vals or 'Skim-Skip' in reading_vals:
        return 'review'
    if 'Not-Started' in status_vals or 'To-Solve' in status_vals or 'To-Read' in reading_vals or str(fm.get('Studied')).lower() == 'false':
        return 'todo'
    if status_vals or reading_vals or ('Studied' in fm) or ('Quiz taken' in fm):
        return 'triage'
    return None


def describe(fm):
    parts = []
    if fm.get('Status') is not None:
        parts.append('Status: ' + ', '.join(listv(fm.get('Status'))))
    if fm.get('Reading Status') is not None:
        parts.append('Reading: ' + ', '.join(listv(fm.get('Reading Status'))))
    if fm.get('Due Date') is not None:
        parts.append('due ' + str(fm.get('Due Date')).strip('"'))
    elif fm.get('Date') is not None:
        parts.append('date ' + str(fm.get('Date')).strip('"'))
    if fm.get('Tier') is not None:
        parts.append('tier ' + ', '.join(listv(fm.get('Tier'))))
    if fm.get('Category') is not None:
        parts.append('cat ' + ', '.join(listv(fm.get('Category'))[:3]))
    elif fm.get('Topic') is not None:
        parts.append('topic ' + ', '.join(listv(fm.get('Topic'))[:2]))
    return ' · '.join(parts)


def sort_key(card):
    # Due dates first, then scheduled Date, then path.
    due = card.get('due') or dt.date.max
    date = card.get('date') or dt.date.max
    return (due, date, card['path'])


def collect_status_cards():
    cards = []
    for md in sorted(ROOT.rglob('*.md')):
        if not should_include(md):
            continue
        fm, _body = parse_fm(md.read_text(errors='ignore'))
        lane = classify(fm)
        if not lane:
            continue
        due = parse_date(fm.get('Due Date'))
        date = parse_date(fm.get('Date')) or parse_date(fm.get('Assigned Date')) or parse_date(fm.get('Date Added'))
        desc = describe(fm)
        area = str(md.relative_to(ROOT)).split('/', 1)[0]
        area_tag = {
            '01 Learning': '#area-learning',
            '02 Research': '#area-research',
            '03 Algorithms': '#area-algorithms',
            '04 Reference': '#area-reference',
        }.get(area, '#area-other')
        cards.append({
            'lane': lane,
            'path': str(md.relative_to(ROOT)),
            'title': md.stem,
            'link': wikilink(md),
            'desc': desc,
            'due': due,
            'date': date,
            'area_tag': area_tag,
            'mtime': md.stat().st_mtime,
        })
    cards.sort(key=sort_key)
    return cards


def cards_by_lane(cards):
    lanes = defaultdict(list)
    for card in cards:
        lanes[card['lane']].append(card)
    return lanes


def card_to_markdown(card):
    desc = f' — {card["desc"]}' if card['desc'] else ''
    due = f' 📅 {card["due"].isoformat()}' if card.get('due') else ''
    return f'- {card["link"]}{desc}{due} {card["area_tag"]}'


if __name__ == '__main__':
    cards = collect_status_cards()
    counts = Counter(card['lane'] for card in cards)
    print(f'Collected {len(cards)} status cards: ' + ', '.join(f'{k}={counts[k]}' for k in sorted(counts)))
