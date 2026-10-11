# Tokenization bugs in the reward-model scoring path: findings

Model under study: `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`. Written on the `kv-cached` branch
(August 2026); paths below are updated to the current layout.

## Summary

The original vocabulary-sweep scoring path (`RewardModel.get_reward_scores_from_response_token_ids`
in `core/reward_model_support.py`) has two tokenization issues. Building a KV-cached scoring path
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

**Update (October 2026): checked at full scale.** Across all 245 (template, persona)
combinations (`kv_cache/main_vs_kv_persona_shift.ipynb`, output in
`kv_cache/output/persona_shift_main_vs_kv_cached.csv`):

| | value |
|---|---|
| corr(τ_main, τ_kv) across combinations | 0.977 |
| corr(RBO_main, RBO_kv) across combinations | 0.868 |
| mean Δτ = τ_kv − τ_main | +0.035 |
| share of combinations with Δτ > 0 | 95.5% |
| mean ΔRBO | +0.014 (56% > 0) |

So the direction holds at scale: the main path reports slightly *more* persona shift than the
fixed path in almost every combination. The relative ordering of personas still agrees closely
on τ, less closely on RBO.

## Bottom line

- **Absolute rankings are substantially wrong** under the original pipeline for this model —
  Table 2-style "here's the literal top token" claims need re-derivation from the fixed pipeline.
- **Comparative/shift claims (which persona shifts more/less) look much more robust** — the
  8-persona sample shows the relative story replicating closely, with a possible small systematic
  bias worth checking at scale.
- Both issues predate this branch (present since `e82618b`, the initial public release) and
  affect the original `paper/generate_reward_model_scores.py` / `persona/generate_persona_reward_model_scores.py`
  equally — this isn't specific to the persona-analysis extension.

## A third mechanism: the chat template trims the response

Found in the October 2026 code review. The GRM (Llama 3.2) chat template renders every message
as `message['content'] | trim`, so on the main path a decoded response like `" freedom"` is
templated as `"freedom"`: the leading space that distinguishes the two tokens is stripped before
re-tokenization. In `data/reward_model_scores/`, 9,703 of the 18,378 `" word"`/`"word"` pairs
(53%) get *identical* scores on all 7 prompts; in `data/reward_model_scores_kv_cached/`, none do.
This is part of why the round trip fails so often, and it means the main path can't tell
space-prefixed tokens from their bare counterparts. The fixed and kv-cached paths never put the
response through the template as text, so they're unaffected.

## A confound in every run: the date in the system prompt

Also found in the October 2026 review. The Llama 3.x chat template writes
`Today Date: <date>` into the system header, taking the date from the clock at render time
(`strftime_now`, available in transformers 4.43+) unless a `date_string` is passed. None of the
committed scores passed one, so every run on transformers 4.45+ (all of this fork's runs) scored a
slightly different prompt depending on the day: the matched baseline columns and persona sweep
(July 2026), the kv and fixed runs (August 2026) and the MFD runs (August 14, 2026). The paper's
original `greatest`/`best`/`worst` columns may predate `strftime_now`, in which case they used the
template's fallback date. A persona-vs-baseline shift therefore also includes a date change
wherever the two columns were scored on different days. The code now pins `date_string` to
`CHAT_TEMPLATE_DATE` (`"26 Jul 2024"`, the template's own fallback) everywhere it applies a chat
template; the committed data predates that and would need re-scoring to remove the confound.

## Other things found along the way

- `generate_persona_reward_model_scores.py`'s `df.to_csv(...)` call was missing
  `escapechar='\\'` (present in `generate_reward_model_scores.py`) — crashes on any token whose
  decoded string needs CSV escaping. Fixed on this branch. *(October 2026: `escapechar` turned
  out to double every backslash in `token_decoded` on read-back, since the readers don't pass it.
  The generators now write with pandas' default quoting, which needs no escape character, and
  rebuild the token columns from the tokenizer on resume. The affected committed files were
  repaired; see `docs/CODE_REVIEW_2026-10.md`, finding 11.)*
