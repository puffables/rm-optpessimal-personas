# Reward Model Interpretability via Optimal and Pessimal Tokens

Extended from code for the paper: **"Reward Model Interpretability via Optimal and
Pessimal Tokens"** by Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson,
Christopher Summerfield, and Tsvetomira Dumbalska (ACM FAccT 2025).
[Read the paper.](https://dl.acm.org/doi/full/10.1145/3715275.3732068)

This repository ranks tokens by their reward scores, revealing biases and
interpretability insights. It then adds scaffolding to do the above for
persona-appended prompts ("I am [x]. {QUERY}") and differently-framed prompts.

## Setup

```bash
conda create -n reward-model-tokens python=3.10
conda activate reward-model-tokens
pip install -r requirements.txt
```

You will also need a [HuggingFace](https://huggingface.co/) account with access to
the relevant model weights.

Figures 3, A3, and A4 are generated with R. Install [R](https://www.r-project.org/)
and the required packages:

```r
install.packages(c("tidyverse", "tidytext", "textdata", "ggpubr", "broom", "ggh4x", "ggbeeswarm"))
```

## Data

Data for the analyses is included under `data/`:

| Directory | Description |
|-----------|-------------|
| `data/reward_model_scores/` | Per-model neutral (no-persona) reward score CSVs. Columns: `token_id, token_name, token_decoded`, then one column per baseline prompt: `greatest, best, worst, greatest_plain, greatest_i_think, greatest_you_think, greatest_people_think` |
| `data/persona_reward_model_scores/` | Persona-conditioned reward scores (output of `generate_persona_reward_model_scores.py`). One column per `template__persona` combination. Full CSVs are reconstructed from `parts/` by `reassemble.py` (see below) |
| `data/base_model_logits/` | Base-model next-token logits for the neutral prompts (output of `generate_base_model_logprobs.py`) |
| `data/persona_base_model_logits/` | Base-model logits for persona-conditioned prompts (output of `generate_persona_base_model_logprobs.py`) |

The `data/corpora/` (AFINN/Bing sentiment, reference corpora) and `data/elo/`
(EloEverything scores) inputs used by some of the original paper's R figures are
**not bundled** in this fork; the sentiment corpora are loaded on demand via
`tidytext` / `textdata`.

## Usage

### Generate reward model scores

Score every token in each model's vocabulary against each prompt:

```bash
python generate_reward_model_scores.py
```

Models and prompts are configured in `config/reward_models.yaml` and
`config/prompts.yaml`. Output CSVs are saved to `data/`.

> **Note:** `config/reward_models.yaml` currently has a single reward model
> active (`Ray2333/GRM-Llama3.2-3B`); the other models from the original paper are
> present but commented out. Uncomment the ones you want to run.

### Core modules

- `reward_model_support.py` — Base `RewardModel` class with factory pattern and device management
- `reward_model_registry.py` — Registry of 10 reward models (Llama 3 and Gemma 2 families)
- `analysis_support.py` — Correlation metrics, vocabulary operations, and visualization helpers

### Generate figures and tables

```bash
python figures/generate_figure_1.py
python figures/generate_figure_2.py
Rscript figures/generate_figure_3.R
python figures/generate_figure_4.py
python figures/generate_figure_5.py

python tables/generate_table_1.py
python tables/generate_table_2.py
python tables/generate_table_3.py
python tables/generate_tables_a1_through_a5.py
python tables/generate_tables_a6_through_a9.py
```

### Multi-token search using Greedy Coordinate Gradient (GCG)

For code relating to multi-token search using a custom implementation of nanoGCG,
see https://github.com/thompsonj/nanoGCG.

### Base model log-probabilities

Generate base model log-probabilities (required by Tables A1–A5):

```bash
python generate_base_model_logprobs.py
```

Configuration: `config/gemma_base_models.yaml` and `config/prompts.yaml`.

## Persona extension

Scaffolding added on top of the original paper to measure how persona
self-identification ("I am {persona}. …") shifts a reward model's token rankings.

**Configuration**
- `config/personas.yaml` — 49 personas across race, gender, intersectional, age,
  disability, religion, and political axes, plus a `control` persona ("a person")
  that isolates the effect of any prefix from the effect of a specific identity.
- `config/persona_prompts.yaml` — prompt templates, each mapping to a matched
  neutral `baseline_column` and a `group` (framings of the same question).

**Generation**
```bash
python generate_persona_reward_model_scores.py    # reward scores per template/persona
python generate_persona_base_model_logprobs.py    # base-LM logits per template/persona
```
Both checkpoint after every column and are resumable — re-running skips
already-scored combinations.

**Analysis**
```bash
python persona_analysis/analyze_personas.py       # τ / ρ / RBO vs. baseline, rank shifts → persona_analysis/output/
python persona_analysis/build_dashboard.py        # interactive HTML dashboard
```
`personaFigures/gen_figures.ipynb` renders the persona figures from the
`persona_analysis/output/` CSVs. A `SETUP/CONFIG` cell at the top exposes three
switches — `PROMPT_TEMPLATE` (`default greatest` / `what do you think` /
`what do I think` / `what do people think` / `worst`), `BASELINE_TYPE`
(`neutral` / `control`), and `RBO_P` — and every figure/table is written to
`personaFigures/output/` with the template and baseline in its filename. The
notebook is organised as five questions:

1. **RM persona shifts** — ranking consistency (Kendall's τ, RBO) vs. baseline by
   persona category, plus each category's least-consistent persona and its
   top/bottom-scoring tokens.
2. **RM vs. base LM** — do the RM's persona shifts track the base model's? A
   coarse correlation, the scatter behind it, and a per-persona "RM-amplified
   shift" breakdown.
3. **Prompt valence** — persona shift under the `greatest` vs. the `worst` prompt.
4. **Agency framing** — how the "what do I / you / people think" reframings relate
   to the default prompt, measured both per persona (4a) and across personas (4b).
5. **Extensions** — the RM↔LM relationship split by baseline, and within category.

The full persona reward-score CSVs exceed GitHub's 100MB limit, so only the split
chunks under `data/persona_reward_model_scores/parts/` are tracked. They are
reconstructed locally on first use — the analysis scripts call
`reassemble.ensure_all()` automatically, or run it directly:

```bash
python data/persona_reward_model_scores/reassemble.py
```

The persona base-model logits (`data/persona_base_model_logits/`) likewise exceed
the 100MB limit and are **not tracked**; regenerate them locally with
`python generate_persona_base_model_logprobs.py`.

## Review notes

A critical review of this fork's methodology, statistics, and implementation is in
[`docs/CRITICAL_REVIEW.md`](docs/CRITICAL_REVIEW.md) (also available as HTML/PDF in
`docs/`).

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
