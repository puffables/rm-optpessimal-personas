"""
Generate reward model scores for all tokens in the vocabulary.

For each active reward model in config/reward_models.yaml, scores every token
in the model's vocabulary against each prompt in config/prompts.yaml, and saves
results to data/reward_model_scores/<model>.csv.

Checkpoints after every prompt column, so an interrupted run — or one where new
prompts were added to config/prompts.yaml after a model was already scored — can
be resumed by simply re-running the script: already-scored prompt columns are
skipped, and only pending ones are computed and appended.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import DATA_DIR, active_reward_models, load_config, safe_model_name
from core.reward_model_support import RewardModel
from core.reward_model_registry import *  # registers all models
from core.score_files import existing_columns, load_checkpoint, save_checkpoint, vocab_frame

DEFAULT_OUTPUT_DIR = DATA_DIR / 'reward_model_scores'

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model names or nicknames to run (default: every entry in '
                          'reward_models.yaml not marked active: false)')
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
parser.add_argument('--fixed', action='store_true',
                     help='Fix the duplicate-BOS and decode-then-re-tokenize round-trip '
                          'issues in the original scoring path, without KV-caching (full '
                          'forward pass per candidate token, same cost as the original — '
                          'use this to isolate the effect of the tokenization fix alone, '
                          'independent of caching). Mutually exclusive with --kv-cache, '
                          'which already includes both fixes.')
parser.add_argument('--output-dir', type=str, default=None,
                     help='Where to write per-model CSVs (default: data/reward_model_scores). '
                          'Override this to compare a --kv-cache/--fixed run against the '
                          'default-path output without the checkpoint-skip logic treating '
                          'already-scored prompt columns as done.')
args = parser.parse_args()
if args.kv_cache and args.fixed:
    raise ValueError("--kv-cache already includes the tokenization fixes --fixed applies "
                      "without caching — pass only one.")
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

prompts = load_config('prompts.yaml')

if args.prompts:
    selected = args.prompts.split(',')
    prompts = {k: v for k, v in prompts.items() if k in selected}

if args.models:
    selected = set(args.models.split(','))
    models = [m for m in load_config('reward_models.yaml')
              if m['name'] in selected or m['nickname'] in selected]
else:
    models = active_reward_models()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for model_info in models:
    model_name = model_info['name']
    safe_name = safe_model_name(model_name)
    output_path = OUTPUT_DIR / f"{safe_name}.csv"

    already_scored = existing_columns(output_path)
    pending = {name: text for name, text in prompts.items() if name not in already_scored}
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
    if args.kv_cache:
        score_fn = reward_model.get_reward_scores_from_response_token_ids_kv_cached
    elif args.fixed:
        score_fn = reward_model.get_reward_scores_from_response_token_ids_fixed
    else:
        score_fn = reward_model.get_reward_scores_from_response_token_ids

    # Build vocabulary (shared across prompts for this model)
    df = load_checkpoint(output_path, vocab_frame(tokenizer))
    token_ids = df['token_id'].tolist()

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
        save_checkpoint(df, output_path)  # checkpoint after every prompt

    print(f"Saved {safe_name}.csv")

    # Memory cleanup
    del reward_model
    torch.cuda.empty_cache()

print("Done.")
