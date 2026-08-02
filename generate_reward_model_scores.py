"""
Generate reward model scores for all tokens in the vocabulary.

For each reward model listed in config/reward_models.yaml, scores every token
in the model's vocabulary against each prompt in config/prompts.yaml, and saves
results to a CSV file in data/.

Checkpoints after every prompt column, so an interrupted run — or one where new
prompts were added to config/prompts.yaml after a model was already scored — can
be resumed by simply re-running the script: already-scored prompt columns are
skipped, and only pending ones are computed and appended.
"""

import argparse
import yaml
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from reward_model_support import RewardModel
from reward_model_registry import *  # registers all models

SCRIPT_ROOT = Path(__file__).parent
CONFIG_DIR = SCRIPT_ROOT / 'config'
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / 'data' / 'reward_model_scores'

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model names or nicknames to run (default: all in reward_models.yaml)')
parser.add_argument('--prompts', type=str, default=None,
                     help='Comma-separated prompt names to run (default: all in prompts.yaml)')
parser.add_argument('--batch-size', type=int, default=None,
                     help='Token batch size for scoring (default: each model\'s configured '
                          'batch_size in reward_models.yaml, tuned conservatively — raise this '
                          'on bigger GPUs, e.g. an A100)')
parser.add_argument('--kv-cache', action='store_true',
                     help='Cache the shared prompt prefix once per prompt instead of '
                          're-tokenizing and re-running it through the model for every '
                          'candidate token (much faster; only supports single-GPU models).')
parser.add_argument('--output-dir', type=str, default=None,
                     help='Where to write per-model CSVs (default: data/reward_model_scores). '
                          'Override this to compare a --kv-cache run against the default-path '
                          'output without the checkpoint-skip logic treating already-scored '
                          'prompt columns as done.')
args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

# Load configs
with open(CONFIG_DIR / 'prompts.yaml') as f:
    prompts = yaml.safe_load(f)

if args.prompts:
    selected = args.prompts.split(',')
    prompts = {k: v for k, v in prompts.items() if k in selected}

with open(CONFIG_DIR / 'reward_models.yaml') as f:
    models = yaml.safe_load(f)

if args.models:
    selected = set(args.models.split(','))
    models = [m for m in models if m['name'] in selected or m['nickname'] in selected]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for model_info in models:
    model_name = model_info['name']
    safe_name = model_name.replace('/', '--')
    output_path = OUTPUT_DIR / f"{safe_name}.csv"

    df = None
    existing_columns = set()
    if output_path.exists():
        df = pd.read_csv(output_path)
        existing_columns = set(df.columns)

    pending = {name: text for name, text in prompts.items() if name not in existing_columns}
    if not pending:
        print(f"Skipping {model_name} — all prompts already scored")
        continue

    print(f"Processing model: {model_name} ({len(pending)} prompts to score)")
    reward_model = RewardModel.create(model_name)
    tokenizer = reward_model.tokenizer

    if args.kv_cache and reward_model.multi_gpu:
        raise ValueError(
            f"{model_name} is configured multi_gpu: true — KV-cached scoring only "
            "supports the single-GPU path. Re-run without --kv-cache for this model."
        )
    score_fn = (reward_model.get_reward_scores_from_response_token_ids_kv_cached
                if args.kv_cache else
                reward_model.get_reward_scores_from_response_token_ids)

    # Build vocabulary (shared across prompts for this model)
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids],
                                           skip_special_tokens=False)

    if df is None:
        df = pd.DataFrame({
            'token_id': token_ids,
            'token_name': token_names,
            'token_decoded': token_decoded,
        })

    # Score each pending prompt
    batch_size = args.batch_size or reward_model.default_batch_size
    for prompt_name, prompt_text in pending.items():
        print(f"  Prompt: {prompt_name}")
        all_scores = []

        for i in tqdm(range(0, len(token_ids), batch_size),
                       desc=f"  {prompt_name}"):
            batch_ids = list(token_ids[i:i + batch_size])
            scores = score_fn(prompt_text, batch_ids, batch_size)
            all_scores.extend(scores)

        df[prompt_name] = all_scores
        df.to_csv(output_path, index=False, escapechar='\\')  # checkpoint after every prompt

    print(f"Saved {safe_name}.csv")

    # Memory cleanup
    del reward_model
    torch.cuda.empty_cache()

print("Done.")
