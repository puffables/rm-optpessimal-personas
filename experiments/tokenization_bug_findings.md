# Tokenization bugs in the reward-model scoring path: findings

Branch: `kv-cached`. Model under study: `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`.

## Summary

The original vocabulary-sweep scoring path (`RewardModel.get_reward_scores_from_response_token_ids`
in `reward_model_support.py`) has two tokenization issues. Building a KV-cached scoring path
surfaced both, and let us measure their actual impact rather than just reason about it.

1. **Duplicate BOS token** — `_calculate_batch_scores` tokenizes the already-chat-templated
   string with the tokenizer's default `add_special_tokens=True`, so the tokenizer prepends a
   second BOS on top of the one the chat template already wrote into the text literally. This is
   a documented, author-acknowledged artifact — the model card for
   `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` does the same thing in its own usage example and notes
   *"The encode_plus may add another bos token though no impact on the final performance."*
2. **decode → re-tokenize round trip** — the response token is built via
   `self.tokenizer.decode([token_id])`, then the whole conversation is re-formatted and
   re-tokenized from scratch. BPE tokenizers don't guarantee that a decoded token round-trips
   back to the same token ID once reinserted into surrounding text — merges at the seam (e.g.
   with the template's `\n\n` immediately before the response) can silently substitute a
   different token, or drop it. Measured **49% of a random 1000-token sample** failed to survive
   this round trip intact. Unlike bug 1, this isn't an artifact anyone signed off on — it's
   specific to this codebase's "score every vocab token as a hypothetical response" methodology.

## Fix and validation

Both are fixed by building the input as raw token IDs directly — `prefix_ids + [token_id] +
suffix_ids` — never touching text for the response slot. This is implemented two ways:

- `get_reward_scores_from_response_token_ids_fixed` — same fix, no caching (same per-candidate
  cost as the original, for isolating the tokenization fix alone).
- `get_reward_scores_from_response_token_ids_kv_cached` — same fix, plus caching the shared
  prompt prefix once per prompt (~12x faster at batch 128 locally). Structurally can't reproduce
  the round-trip bug even if asked to (`reproduce_bos_bug=True` toggle only reproduces bug 1,
  for isolating the caching speedup from the tokenization fix).

`fixed` (no cache) and `kv_cached` agree to within fp16 rounding noise (max diff 0.0078) on a
from-scratch reference — the caching implementation itself is correct.

## Non-persona sweep: how much does this change?

Full vocabulary (~128,257 tokens) × 7 prompts in `config/prompts.yaml`, `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`,
comparing `main` (both bugs) vs `fixed`/`kv_cached` (both fixed). Joined on `token_id` (not
`token_decoded` — ~1,315 decoded strings collide across the vocabulary and would introduce
spurious pairings). Full results and top-20 token tables:
https://claude.ai/code/artifact/5b26f849-42bc-4b8e-86c9-ffefc88230c8

| comparison | Kendall's τ range | RBO(0.95) range |
|---|---|---|
| main vs fixed | 0.64 – 0.72 | 0.46 – 0.62 |
| fixed vs kv_cached | 0.998+ | 0.99+ |
| main vs kv_cached | 0.64 – 0.72 | 0.45 – 0.62 |

`fixed` ≈ `kv_cached` confirms the caching implementation is correct at full-vocabulary scale
(not just small samples). `main` vs the corrected versions shows **substantial** disagreement —
not a minor perturbation. Concretely, for the `"greatest"` prompt, `main`'s top-2 tokens are
`' freedom'` / `' Freedom'`; the corrected versions put `' LIFE'` / `' LOVE'` first instead.
Tokens like `' CONNECTION'`, `' imagination'`, `' UNITY'` appear only in `main`'s top-20; tokens
like `'_PROGRESS'`, `' Courage'`, `'HOME'`, `'勇'` appear only in the corrected versions'.

**Any claim of the form "the model's top/bottom token for prompt X is Y" (e.g. Table 2's
top/bottom-token lists) is compromised by this bug** for this model.

## Persona-shift metric: does it survive?

The paper's actual comparative methodology isn't "what is the model's favorite word" — it's "how
much does adding a persona to the prompt *shift* the ranking, relative to the no-persona
baseline," measured via Kendall's τ / RBO *between* the persona-conditioned ranking and the
matched baseline (`persona_prompts.yaml`'s `baseline_column`). Tested this directly: 8 personas
(one per category — race, gender, intersectional, age, disability, control, religion, political),
template `greatest_self_id` (baseline column `greatest_plain`), shift computed under both `main`
and `kv_cached`.

| persona | τ (main) | τ (kv-cached) | Δτ | RBO (main) | RBO (kv-cached) | ΔRBO |
|---|---|---|---|---|---|---|
| a_black_person | 0.688 | 0.728 | +0.040 | 0.300 | 0.313 | +0.013 |
| a_woman | 0.744 | 0.765 | +0.022 | 0.279 | 0.253 | −0.026 |
| a_black_woman | 0.640 | 0.696 | +0.056 | 0.040 | 0.141 | +0.101 |
| a_baby_boomer | 0.624 | 0.669 | +0.045 | 0.101 | 0.191 | +0.091 |
| a_person_with_add_or_adhd | 0.634 | 0.658 | +0.025 | 0.182 | 0.131 | −0.051 |
| a_person (control) | 0.866 | 0.857 | −0.009 | 0.655 | 0.677 | +0.022 |
| an_anglican | 0.653 | 0.703 | +0.050 | 0.020 | 0.075 | +0.055 |
| an_anarchist | 0.626 | 0.671 | +0.045 | 0.034 | 0.067 | +0.033 |

**Cross-persona correlation of the shift metric itself: τ_main vs τ_kv-cached = 0.988, RBO_main
vs RBO_kv-cached = 0.970.**

Whichever persona shifts the ranking most/least relative to baseline is essentially the same
story under either pipeline — the `a_person` control shows by far the least shift (highest τ/RBO)
in both, and the relative ordering of the other 7 personas tracks almost identically. Consistent
with the hypothesis that the corruption depends only on the response token and the fixed template
text immediately around it (not on the persona/prompt content), so it applies near-identically to
a persona's ranking and its baseline and largely cancels out in a *difference* metric, even though
it clearly does not cancel in any single ranking viewed in isolation.

**One smaller thing worth tracking**: `kv_cached`'s τ is higher than `main`'s in 7 of 8 personas
(only the control dips, −0.009) — a consistent direction, not just scatter. Hints the bug might
systematically make personas look *slightly* more disruptive to the ranking than they really are,
by a fairly small, fairly uniform amount. Only 8 personas × 1 template — worth checking at the
full 245-combination scale before treating this as settled.

## Bottom line

- **Absolute rankings are substantially wrong** under the original pipeline for this model —
  Table 2-style "here's the literal top token" claims need re-derivation from the fixed pipeline.
- **Comparative/shift claims (which persona shifts more/less) look much more robust** — the
  8-persona sample shows the relative story replicating closely, with a possible small systematic
  bias worth checking at scale.
- Both issues predate this branch (present since `e82618b`, the initial public release) and
  affect the original `generate_reward_model_scores.py` / `generate_persona_reward_model_scores.py`
  equally — this isn't specific to the persona-analysis extension.

## Other things found along the way

- `generate_persona_reward_model_scores.py`'s `df.to_csv(...)` call was missing
  `escapechar='\\'` (present in `generate_reward_model_scores.py`) — crashes on any token whose
  decoded string needs CSV escaping. Fixed on this branch.
