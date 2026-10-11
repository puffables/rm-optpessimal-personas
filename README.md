# Reward Model Interpretability via Optimal and Pessimal Tokens: persona extension

Extended from the code for the paper **"Reward Model Interpretability via Optimal and
Pessimal Tokens"** by Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson,
Christopher Summerfield, and Tsvetomira Dumbalska (ACM FAccT 2025).
[Read the paper.](https://dl.acm.org/doi/full/10.1145/3715275.3732068)

The paper scores every token in a reward model's vocabulary as a one-token response to
a prompt like "What, in one word, is the greatest thing ever?", then studies the
resulting rankings. This fork reproduces that and adds four experiments on
`Ray2333/GRM-Llama3.2-3B-rewardmodel-ft`:

| Experiment | Folder | Question |
|---|---|---|
| Paper reproduction | [`paper/`](paper/) | The original paper's scores, figures and tables, across 10 reward models |
| Persona extension | [`persona/`](persona/) | How much does "I am {persona}. ..." shift the reward model's token ranking, and does the base LM shift the same way? |
| KV cache and tokenization | [`kv_cache/`](kv_cache/) | A faster, KV-cached scoring path, and the tokenization bugs in the original path it exposed |
| Moral Foundations Dictionary | [`mfd/`](mfd/) | When a user discloses a moral value ("I value fairness the most..."), does the reward model reward words from that foundation? |
| Vector analysis | [`vector_analysis/`](vector_analysis/) | Are a persona's top-rewarded tokens semantically associated with that persona (word2vec)? |

Known issues and open questions from the October 2026 code review are in
[`docs/CODE_REVIEW_2026-10.md`](docs/CODE_REVIEW_2026-10.md). Read it before relying on
any absolute token rankings from the original scoring path.

## Repository layout

```
core/              shared code: reward-model scoring (3 paths), model registry, paths,
                   ranking helpers (RBO, tie-stable ranking), score-CSV I/O
config/            models, prompts, personas, MFD prompts and substitutions
data/              inputs (dictionaries, corpora, Elo) and every score/logit CSV
paper/             paper reproduction: score generators, figures/, tables/
persona/           persona generators, analysis, dashboard, figures notebook, REPORT.md
kv_cache/          KV-cache experiment, tokenization findings, main-vs-kv comparison
mfd/               Moral Foundations Dictionary experiment notebook and helpers
vector_analysis/   word2vec notebooks
colab/             Colab notebook that runs the GPU steps
docs/              critical review (July 2026) and code review (October 2026)
```

Every script can be run from any directory (`python persona/analyze_personas.py`), and
every notebook finds the repo root itself.

## Setup

```bash
conda create -n reward-model-tokens python=3.10
conda activate reward-model-tokens
pip install -r requirements.txt
playwright install chromium   # for dataframe_image's table PNGs in persona/gen_figures.ipynb
```

Scoring needs a GPU and a [HuggingFace](https://huggingface.co/) token with access to
the model weights (`meta-llama/*` and `google/gemma-*` are gated). The analyses and
notebooks only need the committed CSVs and run on a laptop. To run the GPU steps on
Colab, open [`colab/colab_run_pipeline.ipynb`](colab/colab_run_pipeline.ipynb).

Paper Figures 3, A3 and A4 are generated with R. Install
[R](https://www.r-project.org/) and:

```r
install.packages(c("tidyverse", "tidytext", "textdata", "ggpubr", "broom", "ggh4x", "ggbeeswarm", "yaml"))
```

## Data

Every score/logit CSV has the columns `token_id, token_name, token_decoded`, then one
column per prompt (or per `template__persona` combination). Join on `token_id`: many
distinct tokens decode to the same string.

| Directory | Contents | Produced by |
|---|---|---|
| `data/reward_model_scores/` | Neutral (no-persona) scores for all 10 paper models. Columns `greatest, best, worst` (paper prompts) plus, for R-Lla-3B, the matched persona baselines `greatest_plain, greatest_i_think, greatest_you_think, greatest_people_think` | `paper/generate_reward_model_scores.py` |
| `data/reward_model_scores_fixed/` | R-Lla-3B neutral scores via `--fixed` (tokenization bugs fixed, no cache) | same, `--fixed` |
| `data/reward_model_scores_kv_cached/` | R-Lla-3B neutral scores via `--kv-cache` (bugs fixed, cached) | same, `--kv-cache` |
| `data/persona_reward_model_scores/` | R-Lla-3B persona scores, original path: 49 personas x 5 templates | `persona/generate_persona_reward_model_scores.py` |
| `data/persona_reward_model_scores_kv_cached/` | The same 245 columns via `--kv-cache` | same, `--kv-cache` |
| `data/base_model_logits/` | Next-token logits for the neutral prompts: 10 Gemma base models and `Llama-3.2-3B-Instruct` | `paper/generate_base_model_logprobs.py` |
| `data/persona_base_model_logits/` | `Llama-3.2-3B-Instruct` logits for the 245 persona prompts | `persona/generate_persona_base_model_logprobs.py` |
| `data/mfd_reward_model_scores/` | MFD 2.0 dictionary scored under 96 MFD prompts, kv-cached path, plus a `__runs.csv` describing each column | `mfd/mfd_experiment.ipynb` |
| `data/mfd_reward_model_scores_classic/` | The same 96 columns via the original ("classic") path | same |
| `data/dictionaries/` | MFD 2.0 (`mfd2.0.dic`, typed `mfd2.0_typed.csv`, summary), the R-Lla-3B token vocabulary, EloEverything concepts | `paper/generate_dictionaries.py`, `mfd/generate_typed_dictionary.py` |
| `data/corpora/`, `data/elo/` | BNC word frequencies; EloEverything human rankings (paper Figures 3 and 5) | inputs |

**Large files.** The three persona CSVs are 260 to 325MB each, over GitHub's 100MB
limit, so they're tracked as gzipped copies (`<model>.csv.gz`, about 60MB each). pandas
reads them directly and nothing needs unpacking. The generators write a plain `.csv`
while running (gitignored) and a fresh `.csv.gz` when a model finishes; commit the
`.csv.gz`. Analyses read whichever of the two is newer.

## 1. Paper reproduction (`paper/`)

```bash
python paper/generate_reward_model_scores.py     # reward scores (GPU)
python paper/generate_base_model_logprobs.py     # Gemma base-model logits (GPU; Tables A1 to A5)
python paper/generate_dictionaries.py            # token vocabulary + Elo concept CSVs (CPU)

python paper/figures/generate_figure_1.py        # ... figure_2, 4, 5, a1, a2, a5, a6, a7
Rscript paper/figures/generate_figure_3.R        # ... figure_a3, a4
python paper/tables/generate_table_1.py          # ... table_2, table_3, tables_a1_through_a5, tables_a6_through_a9
```

`config/reward_models.yaml` lists all 10 paper models. Nine are marked `active: false`,
so the generators only score R-Lla-3B unless you name others with `--models`; the
figures and tables always use all 10 models' committed scores. Outputs go to
`paper/figures/output/` and `paper/tables/output/`.

For multi-token search with Greedy Coordinate Gradient (GCG), see the authors'
[nanoGCG fork](https://github.com/thompsonj/nanoGCG).

## 2. Persona extension (`persona/`)

Measures how persona self-identification ("I am {persona}. ...") shifts the reward
model's token ranking, against two baselines: the matched **neutral** prompt with no
persona, and a **control** persona ("a person") that separates the effect of any
"I am X" prefix from the effect of a specific identity.

- `config/personas.yaml`: 49 personas across race, gender, intersectional, age,
  disability, religion and political axes, plus the control.
- `config/persona_prompts.yaml`: 5 templates, each with its matched neutral
  `baseline_column` and a `group` (framings of the same question).

```bash
python persona/generate_persona_reward_model_scores.py           # GPU; add --kv-cache for the fixed path
python persona/generate_persona_base_model_logprobs.py           # GPU; base-LM logits
python persona/analyze_personas.py --variant main                # tau / rho / RBO vs baseline -> persona/output/main/
python persona/analyze_personas.py --variant kv_cached           #   ... on the kv-cached scores -> persona/output/kv_cached/
python persona/build_dashboard.py [--variant kv_cached]          # interactive HTML dashboard
```

Both generators checkpoint after every column and resume where they left off.
`persona/watch_and_refresh.py` reruns the analysis and dashboard as a sweep progresses.

[`persona/gen_figures.ipynb`](persona/gen_figures.ipynb) renders the figures. Its
set-up cell has four switches: `VARIANT` (`main` / `kv_cached`), `PROMPT_TEMPLATE`,
`BASELINE_TYPE` (`neutral` / `control`) and `RBO_P`. Figures go to
`persona/figures/<VARIANT>/`, organised as five questions:

1. **RM persona shifts:** ranking consistency (Kendall's tau, RBO) vs baseline by
   persona category, and each category's least consistent persona's top/bottom tokens.
2. **RM vs base LM:** do the reward model's persona shifts track the base model's?
3. **Prompt valence:** persona shift under `greatest` vs `worst`.
4. **Agency framing:** how "what do I / you / people think" relate to the default prompt.
5. **Extensions:** the RM vs LM relationship split by baseline, and within category.

[`persona/REPORT.md`](persona/REPORT.md) writes up the main-variant results.

## 3. KV cache and tokenization (`kv_cache/`)

`core/reward_model_support.py` has three scoring paths, selected with flags on every
generator:

| Path | Flag | Notes |
|---|---|---|
| original | (none) | The paper's path. Decodes each token to text and re-tokenizes the whole conversation |
| fixed | `--fixed` | Builds inputs from token IDs; same cost as original |
| kv-cached | `--kv-cache` | Same fix, plus the shared prompt prefix is encoded once; ~12x faster |

Building the kv path exposed tokenization bugs in the original one (a duplicate BOS
token, a decode/re-tokenize round trip that substitutes a different token for about half
the vocabulary, and a chat template that strips leading spaces). They change absolute
token rankings substantially but persona *shifts* much less.
[`kv_cache/tokenization_bug_findings.md`](kv_cache/tokenization_bug_findings.md) has the
details.

```bash
python kv_cache/kv_cache_experiment.py --num-tokens 500    # GPU; timing + parity of the three paths
```

[`kv_cache/main_vs_kv_persona_shift.ipynb`](kv_cache/main_vs_kv_persona_shift.ipynb)
compares every persona shift under the original and kv-cached paths (CPU; output in
`kv_cache/output/`).

## 4. Moral Foundations Dictionary (`mfd/`)

[`mfd/mfd_experiment.ipynb`](mfd/mfd_experiment.ipynb) scores every MFD 2.0 dictionary
entry as the assistant's response, under prompts in which the user discloses a moral
value ("I think {substitution} is the greatest thing ever. What do you think...?"), and
analyses whether the disclosed foundation's words move up the ranking (median rank by
category, a control contrast, direct echo sycophancy, top-k tables). It runs locally or
on Colab; its set-up cell picks the scoring path (`SCORING_MODE`), dictionary, templates
(`config/mfd_prompts.yaml`) and substitutions (`config/mfd_substitutions.yaml`).
Analysis outputs go to `mfd/output/<SCORING_MODE>/`.

Helpers (only needed to regenerate committed inputs; need NLTK WordNet):

```bash
python mfd/generate_typed_dictionary.py    # labels MFD 2.0 entries by word type -> data/dictionaries/mfd2.0_typed.csv
python mfd/sample_mfd_substitutions.py     # re-draws the sampled substitutions in config/mfd_substitutions.yaml
```

## 5. Vector analysis (`vector_analysis/`)

- [`vector_analysis.ipynb`](vector_analysis/vector_analysis.ipynb): word2vec cosine
  similarity between every persona and every dictionary token
  (output `vector_analysis/output/cosine_similarity/`).
- [`top_token_semantics.ipynb`](vector_analysis/top_token_semantics.ipynb): tests
  REPORT.md's observation that a persona's top-rewarded tokens look on-theme, by
  comparing each token's similarity to its own persona vs other personas, for either
  scoring variant (output `vector_analysis/output/semantic_association/<variant>/`).

Both download `word2vec-google-news-300` (about 1.6GB) on first run.

## Citation

```bibtex
@inproceedings{christian2025reward,
  title={Reward Model Interpretability via Optimal and Pessimal Tokens},
  author={Christian, Brian and Kirk, Hannah Rose and Thompson, Jessica A.F. and Summerfield, Christopher and Dumbalska, Tsvetomira},
  booktitle={Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year={2025},
  doi={10.1145/3715275.3732068}
}
```

## License

MIT License. See [LICENSE](LICENSE).
