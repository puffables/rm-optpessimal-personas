"""
Watch a persona reward-model scores CSV and refresh the analysis outputs +
dashboard every time a new persona finishes all of its tracked templates
("a layer" of the sweep), without touching the generation script itself.

Polls the CSV header (cheap — doesn't load the 128k-row body) every
--interval seconds, and re-runs analyze_personas.py + build_dashboard.py
whenever the count of "fully-scored" personas increases.

If --watch-pid is given, exits (after one final refresh) once that process
is no longer running — pass the PID of the generate_persona_reward_model_scores.py
run this is watching.
"""

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / 'config'

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str,
                     default='data/persona_reward_model_scores/Ray2333--GRM-Llama3.2-3B-rewardmodel-ft.csv')
parser.add_argument('--templates', type=str,
                     default='greatest_self_id,greatest_self_id_think,greatest_self_id_you_think,greatest_self_id_people_think',
                     help='Comma-separated template names that define a "complete layer" for a persona')
parser.add_argument('--interval', type=int, default=60, help='Poll interval in seconds')
parser.add_argument('--watch-pid', type=int, default=None,
                     help='Exit (after a final refresh) once this PID is no longer running')
args = parser.parse_args()

csv_path = ROOT / args.csv
templates = args.templates.split(',')

with open(CONFIG_DIR / 'personas.yaml') as f:
    personas = yaml.safe_load(f)['personas']


def slugify(text):
    import re
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def pid_alive(pid):
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def completed_personas():
    if not csv_path.exists():
        return set()
    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    complete = set()
    for p in personas:
        slug = slugify(p['name'])
        if all(f"{t}__{slug}" in cols for t in templates):
            complete.add(slug)
    return complete


def refresh():
    print("Refreshing analysis + dashboard...", flush=True)
    subprocess.run(['python', 'persona_analysis/analyze_personas.py'], cwd=ROOT, check=False)
    subprocess.run(['python', 'persona_analysis/build_dashboard.py'], cwd=ROOT, check=False)
    print("Refresh done.", flush=True)


last_complete = completed_personas()
print(f"Watching {csv_path.name} — {len(last_complete)} persona(s) already complete "
      f"across {len(templates)} templates.", flush=True)

while True:
    time.sleep(args.interval)

    current = completed_personas()
    if len(current) > len(last_complete):
        newly = sorted(current - last_complete)
        print(f"New layer(s) complete: {newly}", flush=True)
        refresh()
        last_complete = current

    if args.watch_pid is not None and not pid_alive(args.watch_pid):
        print(f"PID {args.watch_pid} no longer running — doing a final refresh and exiting.", flush=True)
        refresh()
        break

print("Done.")
