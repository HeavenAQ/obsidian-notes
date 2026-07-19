#!/usr/bin/env python3
"""Unified task automation for the Obsidian Kanban board.

This is the source-of-truth board generator used by the local
`claudian-productivity` plugin. It combines:

1. Explicit/inferred deadline lanes from deadline_triage.py.
2. Notion-style Obsidian page status lanes from status_board.py.

Changing a page property such as `Status`, `Reading Status`, `Studied`, or
`Quiz taken` changes the generated Kanban lane the next time this script runs.
"""
import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import deadline_triage
import status_board

ROOT = Path.cwd()
KANBAN = ROOT / '00 Home/Tasks/Task Kanban Board.md'
TRIAGE = ROOT / '00 Home/Tasks/Deadline Triage.md'

MANUAL_LANE_DEFAULTS = {
    '📥 Inbox': [
        '- [ ] [[00 Home/Tasks/Task Inbox]] — Process loose captured tasks #task/inbox',
        '- [ ] [[00 Home/Tasks/Task Command Center]] — Triage tasks with no due/scheduled/start date #task/admin',
        f'- [ ] [[00 Home/Daily Notes/{dt.date.today().isoformat()}]] — Review today’s Top 3 #task/admin',
    ],
    '🎯 Next': [
        '- [ ] [[00 Home/Tasks/Deadline Triage]] — Work through surfaced overdue/deadline items #deadline-triage',
        '- [ ] [[02 Research/Research Thesis]] — Discuss two-layer plan and ML-heavy methods with Prof. Sekine #task/research',
        '- [ ] [[01.3 DL Homework Practice — MIT 6.7960/Homework 1]] — Continue HW1 prep and implementation plan #task/homework',
    ],
    '🚧 Doing': [],
    '🧱 Waiting / Blocked': [
        '- [ ] [[02 Research/Research Thesis]] — Secure dataset access and confirm ethics/usage approvals #task/waiting #task/research',
    ],
    '✅ Manual Done': [
        '- [x] [[00 Home/Tasks/Productivity Plugin Setup]] — Productivity/task stack installed #task/admin',
        '- [x] [[00 Home/Tasks/Task Workflow]] — Task workflow documented #task/admin',
    ],
    'Archive': [],
}

GENERATED_PREFIXES = (
    '📌 Page To Do / Not Started',
    '🚧 Page In Progress / Reading',
    '🔁 Page Review / Skim',
    '✅ Page Done / Read',
    '🧭 Page Needs Triage',
    '🚨 Deadline Overdue',
    '📅 Deadline Today',
    '🗓️ Deadline Upcoming',
)

TAG_COLORS = [
    ('#task/inbox', 'rgba(120, 119, 116, 0.16)'),
    ('#task/admin', 'rgba(203, 145, 47, 0.18)'),
    ('#task/research', 'rgba(51, 126, 169, 0.16)'),
    ('#task/homework', 'rgba(68, 131, 97, 0.16)'),
    ('#task/waiting', 'rgba(212, 76, 71, 0.14)'),
    ('#deadline-triage', 'rgba(217, 115, 13, 0.18)'),
    ('#deadline-homework-plan', 'rgba(68, 131, 97, 0.16)'),
    ('#deadline-homework', 'rgba(203, 145, 47, 0.18)'),
    ('#deadline-dsa', 'rgba(144, 101, 176, 0.16)'),
    ('#deadline-cv', 'rgba(68, 131, 97, 0.16)'),
    ('#deadline-dl-daily', 'rgba(217, 115, 13, 0.16)'),
    ('#deadline-explicit', 'rgba(212, 76, 71, 0.14)'),
    ('#area-learning', 'rgba(68, 131, 97, 0.14)'),
    ('#area-research', 'rgba(51, 126, 169, 0.14)'),
    ('#area-algorithms', 'rgba(144, 101, 176, 0.14)'),
    ('#area-reference', 'rgba(120, 119, 116, 0.14)'),
]


def strip_settings(text):
    # Remove both valid Kanban settings comments and any malformed single-%
    # settings left by older generator runs.
    return re.split(r'\n%{1,2}\s*kanban:settings', text, maxsplit=1)[0].rstrip()


def section_map(text):
    """Return a map from Kanban H2 heading to raw list lines in that lane."""
    text = strip_settings(text)
    matches = list(re.finditer(r'^##\s+(.+?)\s*$', text, re.M))
    sections = {}
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip('\n')
        lines = [line.rstrip() for line in body.splitlines()]
        # Keep list/checklist cards and comments; drop blank-only bodies.
        sections[heading] = lines
    return sections


def normalize_heading(heading):
    return re.sub(r'\s*\(\d+\)\s*$', '', heading).strip()


def clean_lane_lines(lines):
    """Keep only actual Kanban card/list lines from a manual lane."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {'***', '---'}:
            continue
        if stripped.startswith('%') or stripped.startswith('```') or stripped.startswith('{'):
            continue
        if stripped.startswith('-'):
            cleaned.append(line.rstrip())
    return cleaned


def current_manual_lanes(today):
    defaults = {k: list(v) for k, v in MANUAL_LANE_DEFAULTS.items()}
    defaults['📥 Inbox'] = [
        '- [ ] [[00 Home/Tasks/Task Inbox]] — Process loose captured tasks #task/inbox',
        '- [ ] [[00 Home/Tasks/Task Command Center]] — Triage tasks with no due/scheduled/start date #task/admin',
        f'- [ ] [[00 Home/Daily Notes/{today.isoformat()}]] — Review today’s Top 3 #task/admin',
    ]
    if not KANBAN.exists():
        return defaults
    sections = section_map(KANBAN.read_text(errors='ignore'))
    manual = defaults
    for heading, lines in sections.items():
        key = normalize_heading(heading)
        if key in manual and not key.startswith(GENERATED_PREFIXES):
            # Preserve the user's manual Kanban cards exactly as written.
            manual[key] = clean_lane_lines(lines)
        elif key == '✅ Done':
            # Previous board used this name; migrate it to an explicitly manual done lane.
            manual['✅ Manual Done'] = clean_lane_lines(lines)
    return manual


def deadline_card(item):
    return f'- {item["text"]} 📅 {item["due"].isoformat()} {item["tag"]}'


def build_deadline_lanes(deadline_items):
    by = {'overdue': [], 'today': [], 'upcoming': []}
    for item in deadline_items:
        by[item['status']].append(item)
    for key in by:
        by[key].sort(key=lambda x: (x['due'], x['group'], x['text']))
    return by


def build_status_lanes(status_cards):
    by = status_board.cards_by_lane(status_cards)
    order = ['todo', 'doing', 'review', 'done', 'triage']
    return {lane: by.get(lane, []) for lane in order}


def kanban_settings():
    settings = {
        'kanban-plugin': 'board',
        'list-collapse': [],
        'show-checkboxes': True,
        'new-note-folder': '00 Home/Tasks',
        'date-format': 'YYYY-MM-DD',
        'date-display-format': 'YYYY-MM-DD',
        'link-date-to-daily-note': True,
        'tag-colors': [{'tagKey': tag, 'color': color} for tag, color in TAG_COLORS],
    }
    return '%% kanban:settings\n```\n' + json.dumps(settings, ensure_ascii=False, separators=(',', ':')) + '\n```\n%%'


def add_lane(lines, title, cards, empty='- Nothing here.'):
    lines += [f'## {title}', '']
    if cards:
        lines.extend(cards)
    else:
        lines.append(empty)
    lines.append('')


def write_board(deadline_items, status_cards, today):
    manual = current_manual_lanes(today)
    deadline_by = build_deadline_lanes(deadline_items)
    status_by = build_status_lanes(status_cards)

    status_counts = Counter(card['lane'] for card in status_cards)
    deadline_counts = Counter(item['status'] for item in deadline_items)

    lines = [
        '---',
        'kanban-plugin: board',
        'cssclasses:',
        '  - dashboard',
        'Status: In-Progress',
        'Category:',
        '  - Task-Management',
        'Last automated refresh: ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        '---',
        '',
        '<!-- automation-owned: generated by .obsidian/automation/task_board_automation.py -->',
        '<!-- Edit page properties (Status / Reading Status / Studied / Quiz taken) to move generated page cards. -->',
        '',
    ]

    for heading in ['📥 Inbox', '🎯 Next', '🚧 Doing', '🧱 Waiting / Blocked']:
        add_lane(lines, heading, manual.get(heading, []), empty='- Drop manual cards here.')

    status_titles = {
        'todo': '📌 Page To Do / Not Started',
        'doing': '🚧 Page In Progress / Reading',
        'review': '🔁 Page Review / Skim',
        'done': '✅ Page Done / Read',
        'triage': '🧭 Page Needs Triage',
    }
    for lane in ['todo', 'doing', 'review', 'done', 'triage']:
        cards = [status_board.card_to_markdown(card) for card in status_by.get(lane, [])]
        # Do not write an empty triage lane unless it is needed.
        if lane == 'triage' and not cards:
            continue
        add_lane(lines, f'{status_titles[lane]} ({status_counts[lane]})', cards)

    deadline_titles = {
        'overdue': '🚨 Deadline Overdue',
        'today': '📅 Deadline Today',
        'upcoming': '🗓️ Deadline Upcoming',
    }
    for lane in ['overdue', 'today', 'upcoming']:
        cards = [deadline_card(item) for item in deadline_by.get(lane, [])]
        add_lane(lines, f'{deadline_titles[lane]} ({deadline_counts[lane]})', cards)

    add_lane(lines, '✅ Manual Done', manual.get('✅ Manual Done', []), empty='- Move completed manual cards here.')
    archive_cards = manual.get('Archive', [])
    if archive_cards:
        lines += ['***', '', '## Archive', '']
        lines.extend(archive_cards)
        lines.append('')

    lines += [kanban_settings(), '']
    KANBAN.parent.mkdir(parents=True, exist_ok=True)
    KANBAN.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', help='Override today as YYYY-MM-DD')
    parser.add_argument('--skip-triage', action='store_true', help='Do not rewrite Deadline Triage.md')
    args = parser.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

    deadline_items = deadline_triage.collect_items(today)
    status_cards = status_board.collect_status_cards()

    if not args.skip_triage:
        deadline_triage.write_triage(deadline_items, today)
    write_board(deadline_items, status_cards, today)

    deadline_counts = Counter(item['status'] for item in deadline_items)
    status_counts = Counter(card['lane'] for card in status_cards)
    print(
        'Automated task board refreshed: '
        f'deadlines overdue={deadline_counts["overdue"]}, today={deadline_counts["today"]}, upcoming={deadline_counts["upcoming"]}; '
        f'pages todo={status_counts["todo"]}, doing={status_counts["doing"]}, review={status_counts["review"]}, done={status_counts["done"]}, triage={status_counts["triage"]}'
    )


if __name__ == '__main__':
    main()
