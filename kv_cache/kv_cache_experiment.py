"""
Mini experiment: does KV-caching the shared prompt prefix change anything
when scoring many candidate response tokens against the same prompt?

Times and compares three RewardModel scoring paths on the first N vocab tokens:
  - original:  get_reward_scores_from_response_token_ids (decode, re-template
               and re-tokenize the whole conversation per candidate; has the
               double-BOS and round-trip bugs, see tokenization_bug_findings.md)
  - fixed:     get_reward_scores_from_response_token_ids_fixed (both bugs
               fixed, full forward pass per candidate, no caching)
  - kv-cached: get_reward_scores_from_response_token_ids_kv_cached (both bugs
               fixed, prefix encoded once and reused)

fixed vs kv-cached is the parity check: they should agree up to floating-point
noise. original vs either is expected to differ, by the size of the bugs.

Usage:
    python kv_cache/kv_cache_experiment.py --num-tokens 500 --batch-size 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.reward_model_support import RewardModel
from core.reward_model_registry import *  # noqa: F401,F403  (registers model classes)

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

paths = {
    'original': reward_model.get_reward_scores_from_response_token_ids,
    'fixed': reward_model.get_reward_scores_from_response_token_ids_fixed,
    'kv-cached': reward_model.get_reward_scores_from_response_token_ids_kv_cached,
}
scores, times = {}, {}
for name, score_fn in paths.items():
    t0 = time.time()
    scores[name] = torch.tensor(score_fn(PROMPT, candidate_token_ids, args.batch_size))
    times[name] = time.time() - t0
    print(f"{name:>9}: {times[name]:.2f}s  ({num_tokens / times[name]:.1f} tokens/s)")

print(f"\nKV-cache speedup over fixed: {times['fixed'] / times['kv-cached']:.2f}x")


def compare(a, b):
    abs_diff = (scores[a] - scores[b]).abs()
    print(f"\n{a} vs {b}:")
    print(f"  max abs diff:  {abs_diff.max().item():.6f}")
    print(f"  mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"  # differing by > 1e-3: {(abs_diff > 1e-3).sum().item()} / {num_tokens}")
    worst_idx = abs_diff.argmax().item()
    worst_token = reward_model.tokenizer.decode([candidate_token_ids[worst_idx]])
    print(f"  largest-diff token: id={candidate_token_ids[worst_idx]} decoded={worst_token!r} "
          f"({a} {scores[a][worst_idx]:.6f}, {b} {scores[b][worst_idx]:.6f})")


compare('fixed', 'kv-cached')   # parity check: should be ~0
compare('original', 'fixed')    # size of the tokenization bugs
