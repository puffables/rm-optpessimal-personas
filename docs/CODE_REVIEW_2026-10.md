# Code review, October 2026

A read-through of the whole repo (code, configs, notebooks, data and git history) made while
reorganising it into one folder per experiment. It covers correctness, data provenance,
documentation and secrets. The July 2026 [`CRITICAL_REVIEW.md`](CRITICAL_REVIEW.md) covers
methodology and statistics; this review doesn't repeat it.

Each finding is marked:

- **Fixed:** changed in code, and analysis outputs regenerated where that applies.
- **Needs re-scoring:** the code is fixed or the issue is documented, but the committed scores
  still carry it. Removing it needs a GPU rerun.
- **Open:** not addressed yet.

## Summary

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | The chat template stamps the run date into every scored prompt | High | Code fixed; needs re-scoring |
| 2 | `worst_self_id` has no matched neutral baseline | High | Documented; needs re-scoring |
| 3 | Control persona compared against itself inflated control-baseline correlations | High | Fixed |
| 4 | main-vs-kv notes stated the bias direction backwards | High | Fixed |
| 5 | Heavy fp16 ties made RBO and top-k lists depend on sort order | Medium-high | Fixed (deterministic); tie-aware RBO open |
| 6 | Deduplicating before sorting could drop the true top token | Medium | Fixed |
| 7 | Analysis on a fresh clone silently overwrote outputs with empty files | Medium-high | Fixed |
| 8 | MFD outputs didn't record their scoring mode; full-sample run not in the repo | Medium-high | Partly fixed; data gap open |
| 9 | KV cache built through removed attributes on transformers 4.56+ | Medium | Fixed (committed data unaffected) |
| 10 | Chat template trims the response, collapsing `" word"` and `"word"` | Medium | Documented (main path only) |
| 11 | `escapechar` on write doubled backslashes in the token text, breaking 4 paper tables | Medium | Fixed (code and data) |
| 12 | Resuming a run assigned scores by row position | Medium | Fixed |
| 13 | Paper figures/tables broke or shrank to one model | Medium | Fixed |
| 14 | Broken paths after earlier moves; Colab notebooks on three different branches | Medium | Fixed |
| 15 | Duplicated code (kv analysis script and notebook, helpers) | Low | Mostly fixed |
| 16 | Small bugs, dead code, stale comments | Low | Fixed |
| 17 | Vector-analysis method caveats | Medium | Open |

## Findings

### 1. The run date is part of every scored prompt (High; needs re-scoring)

The GRM (Llama 3.2) chat template writes `Today Date: <date>` into the system header from
`strftime_now()` unless `date_string` is passed, and none of the scoring code passed one. So
each run scored a slightly different prompt depending on the day it ran, and a run crossing
midnight changes date mid-column. Every persona-vs-neutral shift compares columns scored on
different days (persona sweep July 10 to 22, 2026; kv runs August 2026), so the shift includes a
date change as well as the persona.

- **Fixed in code:** `core/reward_model_support.py` defines `CHAT_TEMPLATE_DATE = "26 Jul 2024"`
  (the template's own fallback) and passes it in every `apply_chat_template` call, as do both
  base-model logit scripts and the MFD notebook's classic path.
- **To remove the confound from results:** re-score the baselines and persona columns with the
  pinned date. The kv-cached path is the cheapest to rerun.

### 2. `worst_self_id` has no matched baseline (High; needs re-scoring)

`worst_self_id` ("I am {persona}. What is the worst thing ever?") is compared against `worst`
("What, in one word, is the worst thing ever?"), so its neutral-baseline shift also includes
removing "in one word". Every `greatest_*` template has a matched baseline; `worst` never got
one. This affects persona Figure 5 / REPORT.md Section 3 (greatest vs worst valence) and the
`lm_*` worst rows. The control baseline is unaffected, because it uses the same template.

- **Documented** in `config/persona_prompts.yaml` and the dashboard.
- **To fix:** add `worst_plain: What is the worst thing ever?` to `config/prompts.yaml`, score
  it on every path (main, `--fixed`, `--kv-cache`, base LM), and point `worst_self_id` at it.

### 3. Control persona compared against itself (High; fixed)

With `baseline_type == 'control'`, the control persona ("a person") was compared with its own
column, giving rows with tau = RBO = 1. That point sat far from the rest and inflated every
correlation across personas computed under the control baseline (for example, greatest vs
worst tau correlation 0.730 with the row, 0.559 without). Flagged in July as CRITICAL_REVIEW #8.

`persona/analyze_personas.py` no longer emits these rows, and all outputs, figures and
REPORT.md numbers were regenerated.

### 4. main-vs-kv notes had the bias direction backwards (High; fixed)

tau and RBO measure agreement with the baseline, so higher means *less* shift. `d_tau =
tau_kv - tau_main` is positive for 95.5% of combinations, meaning the kv path measures less shift
than main. The notebook and the Colab print said the opposite. `kv_cache/tokenization_bug_findings.md`
had it right. The notebook text is corrected and re-executed.

### 5. fp16 ties (Medium-high; fixed, with a caveat)

The fp16 reward scores are heavily quantised:

| column | distinct values, main path | distinct values, kv path |
|---|---|---|
| `greatest` | 959 | 3,927 |
| `greatest_plain` | 1,516 | 1,711 |
| `worst` | 435 | 2,409 |

out of about 128k tokens each. With pandas' default (unstable) sort, RBO and top-k lists depended
on arbitrary order among tied tokens; comparing a column with itself under two tie orders gave
RBO(p=0.9) as low as 0.90. CRITICAL_REVIEW #9's assumption that "continuous RM scores rarely tie"
doesn't hold.

- **Fixed:** `core.ranking.ranked_ids` breaks ties by `token_id`, so results are deterministic.
  Rerunning moved RBO values by up to 0.02.
- **Open:** deterministic isn't the same as tie-aware. The main path has more ties than the kv
  path, so some of the main-vs-kv RBO gap may be tie noise. A tie-aware RBO, or reporting each
  column's self-RBO ceiling, would settle it.

### 6. Deduplicating before sorting (Medium; fixed)

Top/bottom tables and movers deduplicated on `token_decoded` (keeping the lowest `token_id`)
*before* sorting, so if a higher-id variant of a string was the real top token, it could vanish
from the top 5. `core.ranking.best_per_decoded` sorts first, then keeps each string's best
variant.

### 7. Fresh-clone analysis overwrote outputs with empty files (Medium-high; fixed)

`analyze_personas.py` wrote its CSVs before checking whether anything had been computed, and
skipped missing inputs with only a print. Since several inputs were gitignored, running it on a
fresh clone replaced the committed outputs with empty or half-empty files. Now the large inputs
are tracked (as `.csv.gz`), missing inputs stop the script with a clear message, and nothing is
written unless results exist.

### 8. MFD provenance (Medium-high; partly fixed)

- The analysis cells wrote to one folder whatever `SCORING_MODE` was set to, and the folder was
  gitignored. They now write to `mfd/output/<SCORING_MODE>/`. The existing local outputs were
  checked against both committed score sets (they match the kv-cached scores exactly) and are now
  tracked under `mfd/output/kv_cache/`.
- **Open data gap:** the notebook's saved outputs show a full-sample classic run (726 columns,
  103 substitutions, committed August 14, 2026 as "Full Sample Run completed (no-caching)"),
  but only the notebook was committed. `data/mfd_reward_model_scores_classic/` holds just the
  earlier 96-column run. The full-sample scores exist only in that Colab session's downloads or
  Drive.
- For reference, MFD kv vs classic agree fairly well (Spearman 0.94 to 0.98, median top-10
  overlap 8/10).

### 9. KV cache on newer transformers (Medium; fixed, committed data unaffected)

`get_reward_scores_from_response_token_ids_kv_cached` expanded the prefix cache by assigning
`DynamicCache.key_cache` / `.value_cache`, which transformers 4.56 removed. On newer versions the
assignment succeeds but the model ignores it, silently dropping the prefix: on a tiny random Llama
the old code's scores differed from the no-cache path by up to 0.38, versus 4e-8 after the fix. The
cache is now built through `DynamicCache.update()` (the approach the MFD notebook already used).
All committed kv data came from Colab runs that installed `transformers<4.50`, where the old code
worked, so the data is fine.

### 10. The chat template trims the response (Medium; documented)

The template renders `message['content'] | trim`, so on the main path `" freedom"` is scored as
`"freedom"`. In `data/reward_model_scores/`, 53% of `" word"`/`"word"` pairs get identical scores
on all 7 prompts; in the kv data, none do. Added to `kv_cache/tokenization_bug_findings.md` as a
third mechanism behind the round-trip bug.

### 11. `escapechar` doubled backslashes (Medium; fixed in code and data)

The generators wrote with `escapechar='\\'` and every reader used plain `pd.read_csv`, so each
backslash in `token_name` and `token_decoded` came back doubled. Each time a file was read,
extended and rewritten, those rows got worse. Four committed files were affected (395 tokens
each): `data/reward_model_scores/`, `data/reward_model_scores_fixed/` and
`data/reward_model_scores_kv_cached/` for R-Lla-3B, and
`data/persona_reward_model_scores_kv_cached/`.

This broke paper reproduction. When this fork appended its matched baseline columns to the
R-Lla-3B scores in July 2026, the doubled strings changed which tokens the paper's analyses
treat as shared across models (they match on `token_decoded`), and Tables 3, A7, A8 and A9 no
longer reproduced the committed versions.

- **Code:** the generators now write with pandas' default quoting (no escape character needed)
  and rebuild the token columns from the tokenizer on resume.
- **Data:** the token text in the four files was repaired by undoing the doubling. Before writing,
  this was checked to reproduce exactly the clean text of the same rows: the paper's original
  release for `reward_model_scores`, and the main persona file (never written with
  `escapechar`) for the others. All score columns are byte-for-byte unchanged on re-read.
- **Check:** all 13 paper tables now reproduce the committed outputs exactly, and the 12 Python
  paper figures are pixel-identical. (The three R figures weren't run; R isn't installed on the
  machine used for this review.)

### 12. Resume by row position (Medium; fixed)

Resuming assigned new scores by row position, assuming the CSV on disk had the same rows as the
current vocabulary. It doesn't always: the paper-era CSVs have 128,256 rows and the current
tokenizer has 128,257. `core.score_files.load_checkpoint` now joins existing scores on
`token_id`.

### 13. Paper figures and tables with one active model (Medium; fixed)

With 9 of the 10 models commented out of `config/reward_models.yaml`, several paper scripts
crashed (Table 2, Figures 5 and A7) and the rest silently overwrote the committed 10-model
outputs with 1-model versions. The models are now all listed, with `active: false` on the nine
that the generators should skip.

### 14. Broken paths and branches (Medium; fixed)

Covered by the reorganisation: README and docstring paths, the vector notebooks' cwd-relative
paths (they only worked from the repo root), `kv_cache_experiment.py`'s import, and the four Colab
notebooks, which checked out three different branches (`kv-cached`, `MFD-Run`,
`MFD-and-Analysis`). All now check out `main`. The repo is public, so the clone no longer needs a
GitHub token (one is still used if set).

### 15. Duplication (Low; mostly fixed)

- `kv_analysis/analyze_personas_kv.py` and `kv_analysis/gen_figures_kv.ipynb` were copies of the
  persona script and notebook with two paths changed. Replaced by `--variant` / `VARIANT`.
- `slugify`, `rbo`, `format_token_label`, vocabulary building and checkpoint I/O now live in
  `core/` instead of being copied into each script.
- **Open:** the two reward-score generators (and the two logit generators) still share most of
  their structure, and the MFD notebook has its own multi-token scoring functions. Moving
  `score_prefix_shared` into `core/reward_model_support.py` would make it the single
  implementation, but needs a GPU to verify.

### 16. Small fixes (Low; fixed)

- `analysis_support.plot_multiple_mds` called an undefined function; removed (unused).
- `analyze_personas.py`'s quick-look figures mixed neutral and control rows under a "vs neutral"
  label; now neutral only.
- `kv_cache_experiment.py` reported "score parity" between the original and kv paths, which aren't
  meant to match; it now checks fixed vs kv-cached (should match) and original vs fixed (bug size).
- Unused imports and `python-dotenv` removed; `jupyter`, `ipykernel` and the Playwright step for
  `dataframe_image` added to requirements; R's `yaml` package added to the README.
- Stale comments: "base-LM data doesn't exist yet", "227 combinations" (245), "kv CSVs committed
  whole".
- The MFD notebook carried 5.4MB of saved Colab progress-bar state; stripped from all notebooks.

### 17. Vector-analysis caveats (Medium; open)

Not verified, worth checking before leaning on the word2vec results:

- `DROP_WORDS` removes "with" and "without", so "a person without disabilities" and "a person with
  impaired vision" embed similarly despite opposite meanings. Shared head nouns ("person",
  "woman") also raise within-category similarity.
- A persona's own name among its top tokens (for example "Hindu") would drive the same-persona
  cosine. Excluding name echoes would separate association from echo.

## Things this review didn't change

- **Statistical methodology** (no noise baseline, vacuous p-values, one reward model): see
  CRITICAL_REVIEW #1 to #7.
- **REPORT.md's token-level claims** come from the main scoring path, whose absolute rankings the
  tokenization bugs compromise. Its correlation numbers were updated for finding 3.

## Secrets review

No credentials anywhere in the working tree or the full git history (all branches, 48 commits,
every file version including notebook outputs and metadata). Every notebook loads tokens from
Colab secrets (`userdata.get("HF_TOKEN")`, `userdata.get("GH_TOKEN")`), and no saved output shows
a token. Two minor privacy items:

- A local path containing the author's macOS username appeared in saved outputs of two notebooks.
  Those outputs are cleared; it remains in two older public commits (not worth a history
  rewrite).
- 44 commits carry a personal Gmail as author email. Future commits can use the GitHub noreply
  address instead (`git config user.email`).

## What's gitignored, and why

| Pattern | Reason |
|---|---|
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.DS_Store`, `*.db` | Generated or OS files |
| `.env*` | Secrets belong in Colab secrets or the shell, never the repo |
| `data/persona_*/*.csv` (3 folders) | The plain >100MB checkpoints. The `.csv.gz` copies are tracked |
| `config/gemma_base_models_no27b.yaml` | Scratch config the Colab runner writes |
| `persona/backups/` | Local dated backups |

Deliberately **tracked**: every other score/logit CSV (each under 20MB), analysis outputs and
figures (small, and they let readers see results without rerunning), notebook outputs (minus
widget state), and the persona `.csv.gz` files (about 180MB together). The repo's packed history
is about 200MB already, because the old split parts stay in history; rewriting history to drop
them isn't worth breaking existing clones.
