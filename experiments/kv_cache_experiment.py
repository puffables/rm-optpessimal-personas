"""
Mini experiment: does KV-caching the shared prompt prefix change anything
when scoring many candidate response tokens against the same prompt?

Compares RewardModel.get_reward_scores_from_response_token_ids (baseline,
full re-tokenize + full forward pass per candidate token) against
get_reward_scores_from_response_token_ids_kv_cached (prefix cached once,
only the new token + template suffix run through the model per candidate)
for:
  - wall-clock time
  - score parity (do we get the same numbers back?)

Usage:
    python experiments/kv_cache_experiment.py --num-tokens 500 --batch-size 64
"""

import argparse
import time

import torch

from reward_model_support import RewardModel
from reward_model_registry import *  # noqa: F401,F403  (registers model classes)

MODEL_NAME = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
PROMPT = "i am a doctor. What is the greatest thing ever?"

parser = argparse.ArgumentParser()
parser.add_argument("--num-tokens", type=int, default=500,
                    help="Number of candidate vocab tokens to score (first N by token id)")
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

print(f"Loading {MODEL_NAME} ...")
reward_model = RewardModel.create(MODEL_NAME)

vocab_size = len(reward_model.tokenizer)
num_tokens = min(args.num_tokens, vocab_size)
candidate_token_ids = list(range(num_tokens))
print(f"Scoring {num_tokens} candidate tokens against prompt: {PROMPT!r}\n")

# --- Baseline: full re-tokenize + full forward pass per candidate ---
t0 = time.time()
baseline_scores = reward_model.get_reward_scores_from_response_token_ids(
    PROMPT, candidate_token_ids, args.batch_size)
baseline_time = time.time() - t0
print(f"Baseline:  {baseline_time:.2f}s  ({num_tokens / baseline_time:.1f} tokens/s)")

# --- KV-cached: prefix cached once, only new token(s) run through the model ---
t0 = time.time()
cached_scores = reward_model.get_reward_scores_from_response_token_ids_kv_cached(
    PROMPT, candidate_token_ids, args.batch_size)
cached_time = time.time() - t0
print(f"KV-cached: {cached_time:.2f}s  ({num_tokens / cached_time:.1f} tokens/s)")

print(f"\nSpeedup: {baseline_time / cached_time:.2f}x")

# --- Score parity ---
baseline_t = torch.tensor(baseline_scores)
cached_t = torch.tensor(cached_scores)
abs_diff = (baseline_t - cached_t).abs()

print(f"\nScore parity:")
print(f"  max abs diff:  {abs_diff.max().item():.6f}")
print(f"  mean abs diff: {abs_diff.mean().item():.6f}")
print(f"  # differing by > 1e-3: {(abs_diff > 1e-3).sum().item()} / {num_tokens}")

worst_idx = abs_diff.argmax().item()
worst_token = reward_model.tokenizer.decode([candidate_token_ids[worst_idx]])
print(f"\n  Largest-diff token: id={candidate_token_ids[worst_idx]} decoded={worst_token!r}")
print(f"    baseline score: {baseline_scores[worst_idx]:.6f}")
print(f"    cached score:   {cached_scores[worst_idx]:.6f}")
