"""
Materialize the exhaustive-sweep token vocabularies and the EloEverything
concept list as explicit, standalone CSV files under data/dictionaries/.

Both of these previously existed only implicitly:
  - the token vocabulary was re-derived from tokenizer.get_vocab() every time
    paper/generate_reward_model_scores.py / persona/generate_persona_reward_model_scores.py ran
  - the EloEverything concept list only existed embedded in data/elo/*.csv,
    mixed in with every model's scores across three separate prompt files

Tokenizer loading only needs the tokenizer files (no model weights, no GPU),
so this script has no CUDA dependency.

(The third "dictionary of words and concepts" already exists at
data/corpora/1_1_all_alpha.txt — a BNC-style word frequency list already used
by paper/figures/generate_figure_3.R — so it isn't reproduced here.)
"""
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import DATA_DIR, active_reward_models, safe_model_name
from core.score_files import vocab_frame

OUTPUT_DIR = DATA_DIR / 'dictionaries'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Token dictionaries: one per active model's tokenizer vocabulary ---
for model_info in active_reward_models():
    model_name = model_info['name']
    safe_name = safe_model_name(model_name)
    output_path = OUTPUT_DIR / f"tokens_{safe_name}.csv"

    if output_path.exists():
        print(f"Skipping {model_name} — {output_path} already exists")
        continue

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = vocab_frame(tokenizer)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(df)} tokens)")

# --- EloEverything concept dictionary ---
# All three prompt files in data/elo/ score the exact same 7,530 entities —
# just take the canonical "greatest ever" file's rank/name/score columns.
elo_output_path = OUTPUT_DIR / 'eloeverything_concepts.csv'
if elo_output_path.exists():
    print(f"Skipping EloEverything concepts — {elo_output_path} already exists")
else:
    elo_path = DATA_DIR / 'elo' / 'What one single thing, person, or concept is the greatest ever.csv'
    elo_df = pd.read_csv(elo_path)[['elo_rank', 'name', 'elo_score', 'elo_matches']]
    elo_df.to_csv(elo_output_path, index=False)
    print(f"Saved {elo_output_path} ({len(elo_df)} concepts)")

print("Done.")
