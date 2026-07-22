"""
Reconstruct the full persona reward-model score CSVs from their git-tracked
parts/ chunks. The whole files are too large for GitHub's 100MB limit, so
only the split parts are committed; this rebuilds the originals locally.

Run directly, or import `reassemble(name)` / call `ensure_all()` from other
scripts before reading a CSV in this directory.
"""

import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
PART_RE = re.compile(r'^(?P<base>.+)\.part(?P<idx>\d+)\.csv$')


def _grouped_parts():
    groups = {}
    for p in sorted((HERE / 'parts').glob('*.csv')):
        m = PART_RE.match(p.name)
        if not m:
            continue
        groups.setdefault(m.group('base'), []).append(p)
    return groups


def reassemble(base_name: str) -> Path:
    parts = sorted((HERE / 'parts').glob(f'{base_name}.part*.csv'))
    if not parts:
        raise FileNotFoundError(f'no parts found for {base_name}')

    out_path = HERE / f'{base_name}.csv'
    with open(out_path, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        header = None
        for i, part in enumerate(parts):
            with open(part, newline='') as in_f:
                reader = csv.reader(in_f)
                part_header = next(reader)
                if header is None:
                    header = part_header
                    writer.writerow(header)
                elif part_header != header:
                    raise ValueError(f'header mismatch in {part}')
                for row in reader:
                    writer.writerow(row)
    return out_path


def ensure_all():
    """Reassemble any full CSV that's missing but has parts on disk."""
    for base_name in _grouped_parts():
        out_path = HERE / f'{base_name}.csv'
        if not out_path.exists():
            print(f'Reassembling {out_path.name} from parts...')
            reassemble(base_name)


if __name__ == '__main__':
    for base_name in _grouped_parts():
        print(f'Reassembling {base_name}.csv from parts...')
        reassemble(base_name)
