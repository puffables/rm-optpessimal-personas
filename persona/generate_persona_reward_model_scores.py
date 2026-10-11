"""
Generate reward model scores for persona-conditioned prompts.

For each active reward model in config/reward_models.yaml, and for each
(template, persona) combination defined in config/persona_prompts.yaml and
config/personas.yaml, scores every token in the model's vocabulary and saves
results to data/persona_reward_model_scores/<model>.csv (one column per
template/persona combination, same token_id/token_name/token_decoded format
as data/reward_model_scores/).

Checkpoints after every (template, persona) combination, so an interrupted
run can be resumed by simply re-running the script — already-scored columns
are skipped. When a model finishes, a gzipped copy (<model>.csv.gz) is written
alongside: that's the form tracked in git, since the plain CSV is >100MB.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.paths import DATA_DIR, active_reward_models, load_config, safe_model_name, scores_csv
from core.ranking import slugify
from core.reward_model_support import RewardModel
from core.reward_model_registry import *  # registers all models
from core.score_files import existing_columns, gzip_copy, load_checkpoint, save_checkpoint, vocab_frame

DEFAULT_OUTPUT_DIR = DATA_DIR / 'persona_reward_model_scores'

# Empirically, these small reward models comfortably fit much larger batches
# than the config batch_size (which was tuned conservatively for the original
# single-prompt sweep). ~1024 gave the best throughput/token during smoke
# testing on an RTX Pro 6000 (97GB) with headroom to spare.
PERSONA_BATCH_SIZE = 1024

parser = argparse.ArgumentParser()
parser.add_argument('--templates', type=str, default=None,
                     help='Comma-separated template names to run (default: all in persona_prompts.yaml)')
parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model names or nicknames to run (default: every entry in '
                          'reward_models.yaml not marked active: false)')
parser.add_argument('--personas', type=str, default=None,
                     help='Comma-separated persona slugs to run (default: all in personas.yaml)')
parser.add_argument('--max-personas', type=int, default=None,
                     help='Only run the first N personas (for quick pilots)')
parser.add_argument('--batch-size', type=int, default=PERSONA_BATCH_SIZE,
                     help=f'Token batch size for scoring (default: {PERSONA_BATCH_SIZE}, tuned for a '
                          '97GB GPU — lower this on smaller GPUs, e.g. 128 on a 16GB T4)')
parser.add_argument('--kv-cache', action='store_true',
                     help='Cache the shared prompt prefix once per (template, persona) instead of '
                          're-tokenizing and re-running it through the model for every candidate '
                          'token (much faster; only supports single-GPU models).')
parser.add_argument('--fixed', action='store_true',
                     help='Fix the duplicate-BOS and decode-then-re-tokenize round-trip issues '
                          'without KV-caching (same cost as the original — isolates the '
                          'tokenization fix alone). Mutually exclusive with --kv-cache, which '
                          'already includes both fixes.')
parser.add_argument('--output-dir', type=str, default=None,
                     help='Where to write per-model CSVs (default: data/persona_reward_model_scores). '
                          'Override this to compare a --kv-cache/--fixed run against the '
                          'default-path output without the checkpoint-skip logic treating '
                          'already-scored columns as done.')
args = parser.parse_args()
if args.kv_cache and args.fixed:
    raise ValueError("--kv-cache already includes the tokenization fixes --fixed applies "
                      "without caching — pass only one.")
OUTPUT_DIR = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

if args.models:
    selected = set(args.models.split(','))
    models = [m for m in load_config('reward_models.yaml')
              if m['name'] in selected or m['nickname'] in selected]
else:
    models = active_reward_models()

prompt_templates = load_config('persona_prompts.yaml')['templates']

if args.templates:
    selected = args.templates.split(',')
    prompt_templates = {k: v for k, v in prompt_templates.items() if k in selected}

personas = load_config('personas.yaml')['personas']

if args.personas:
    selected = set(args.personas.split(','))
    personas = [p for p in personas if slugify(p['name']) in selected]

if args.max_personas:
    personas = personas[:args.max_personas]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for model_info in models:
    model_name = model_info['name']
    safe_name = safe_model_name(model_name)
    output_path = OUTPUT_DIR / f"{safe_name}.csv"
    # Resume from the plain checkpoint, or from the tracked .csv.gz on a fresh clone
    resume_path = scores_csv(OUTPUT_DIR, model_name)

    already_scored = existing_columns(resume_path)
    pending = []
    for persona in personas:
        for template_name in prompt_templates:
            col = f"{template_name}__{slugify(persona['name'])}"
            if col not in already_scored:
                pending.append((template_name, persona, col))

    if not pending:
        print(f"Skipping {model_name} — all persona/template combinations already scored")
        continue

    print(f"Processing model: {model_name} ({len(pending)} persona/template combinations to score)")
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

    df = load_checkpoint(resume_path, vocab_frame(tokenizer))
    token_ids = df['token_id'].tolist()

    batch_size = args.batch_size

    for template_name, persona, col in pending:
        prompt_text = prompt_templates[template_name]['text'].format(persona=persona['name'])
        print(f"  {col}: \"{prompt_text}\"")
        all_scores = []

        for i in tqdm(range(0, len(token_ids), batch_size), desc=f"  {col}"):
            batch_ids = list(token_ids[i:i + batch_size])
            scores = score_fn(prompt_text, batch_ids, batch_size)
            all_scores.extend(scores)

        df[col] = all_scores
        save_checkpoint(df, output_path)  # checkpoint after every persona/template

    print(f"Saved {safe_name}.csv")
    print(f"Saved {gzip_copy(output_path).name}")

    del reward_model
    torch.cuda.empty_cache()

print("Done.")
