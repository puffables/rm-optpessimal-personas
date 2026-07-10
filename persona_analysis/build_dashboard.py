"""
Build a self-contained HTML dashboard from persona_analysis/output/summary_correlations.csv.

Re-run any time after analyze_personas.py to refresh with newer data, then
republish via the Artifact tool.
"""
import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / 'output'


def format_token_label(token_decoded):
    """Make a decoded token safe/legible as a table label — see analyze_personas.py's
    copy of this function for why (whitespace-only differences between distinct
    tokens collapse invisibly in HTML otherwise)."""
    if pd.isna(token_decoded):
        return '<NA>'
    s = str(token_decoded)
    if s == '':
        return '<empty>'
    if s.strip(' ') == '':
        return '␣' * len(s)
    leading = len(s) - len(s.lstrip(' '))
    trailing = len(s) - len(s.rstrip(' '))
    core = s[leading:len(s) - trailing] if trailing else s[leading:]
    core = core.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    return ('␣' * leading) + core + ('␣' * trailing)


def safe_read_csv(path, columns):
    """A checkpointed analyze_personas.py run can produce a CSV with no columns at
    all (just a blank line) when nothing was available to score yet (e.g. baseline
    data not generated for any active template) — plain pd.read_csv raises
    EmptyDataError on that, so fall back to an empty frame with the expected
    columns instead of crashing."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


df = safe_read_csv(OUTPUT_DIR / 'summary_correlations.csv', [
    'model', 'model_nickname', 'template', 'persona', 'persona_category', 'n_tokens',
    'kendall_tau', 'kendall_p', 'spearman_rho', 'spearman_p',
    'rbo_p0.90', 'rbo_p0.95', 'rbo_p0.99',
    'lm_available', 'lm_n_tokens', 'lm_kendall_tau', 'lm_kendall_p',
    'lm_spearman_rho', 'lm_spearman_p', 'lm_rbo_p0.90', 'lm_rbo_p0.95', 'lm_rbo_p0.99',
])
rank_shift_df = safe_read_csv(OUTPUT_DIR / 'rank_shifts.csv', [
    'model', 'model_nickname', 'template', 'persona', 'persona_category',
    'direction', 'token_decoded', 'baseline_rank', 'persona_rank', 'rank_diff',
])
top_bottom_df = safe_read_csv(OUTPUT_DIR / 'top_bottom_tokens.csv', [
    'model', 'model_nickname', 'template', 'persona', 'persona_category',
    'group', 'rank_in_group', 'token_decoded', 'score', 'baseline_rank',
])
baseline_top_bottom_df = safe_read_csv(OUTPUT_DIR / 'baseline_top_bottom_tokens.csv', [
    'model', 'model_nickname', 'baseline_column', 'group', 'rank_in_group', 'token_decoded', 'score',
])
template_pair_df = safe_read_csv(OUTPUT_DIR / 'template_pair_correlations.csv', [
    'model', 'model_nickname', 'question_group', 'template_a', 'template_b', 'persona',
    'persona_category', 'n_tokens', 'kendall_tau', 'kendall_p', 'spearman_rho', 'spearman_p',
    'rbo_p0.90', 'rbo_p0.95', 'rbo_p0.99',
])
template_pair_rank_shift_df = safe_read_csv(OUTPUT_DIR / 'template_pair_rank_shifts.csv', [
    'model_nickname', 'question_group', 'template_a', 'template_b', 'persona', 'persona_category',
    'direction', 'token_decoded', 'template_a_rank', 'template_b_rank', 'rank_diff',
])

for _df in (rank_shift_df, top_bottom_df, baseline_top_bottom_df, template_pair_rank_shift_df):
    if 'token_decoded' in _df.columns and not _df.empty:
        _df['token_decoded'] = _df['token_decoded'].apply(format_token_label)

import yaml
ROOT = Path(__file__).parent.parent
with open(ROOT / 'config' / 'persona_prompts.yaml') as f:
    _templates_cfg = yaml.safe_load(f)['templates']
TEMPLATE_TO_BASELINE_COL = {k: v['baseline_column'] for k, v in _templates_cfg.items()}
TEMPLATE_TEXTS = {k: v['text'] for k, v in _templates_cfg.items()}


def detail_key(model_nickname, template, persona):
    return f"{model_nickname}|{template}|{persona}"


def baseline_ref_key(model_nickname, baseline_column):
    return f"{model_nickname}|{baseline_column}"


# Standalone "no persona at all" reference tables, keyed by model + baseline column
baseline_refs = {}
for _, row in baseline_top_bottom_df.iterrows():
    key = baseline_ref_key(row['model_nickname'], row['baseline_column'])
    baseline_refs.setdefault(key, {'top': [], 'bottom': []})
    baseline_refs[key][row['group']].append({'token_decoded': row['token_decoded'], 'score': row['score']})

details = {}
for _, row in df.iterrows():
    key = detail_key(row['model_nickname'], row['template'], row['persona'])
    shifts = rank_shift_df[
        (rank_shift_df['model_nickname'] == row['model_nickname']) &
        (rank_shift_df['template'] == row['template']) &
        (rank_shift_df['persona'] == row['persona'])
    ]
    tb = top_bottom_df[
        (top_bottom_df['model_nickname'] == row['model_nickname']) &
        (top_bottom_df['template'] == row['template']) &
        (top_bottom_df['persona'] == row['persona'])
    ]
    details[key] = {
        'up': shifts[shifts['direction'] == 'up'].head(5)[['token_decoded', 'baseline_rank', 'persona_rank', 'rank_diff']].to_dict(orient='records'),
        'down': shifts[shifts['direction'] == 'down'].head(5)[['token_decoded', 'baseline_rank', 'persona_rank', 'rank_diff']].to_dict(orient='records'),
        'top': tb[tb['group'] == 'top'].sort_values('rank_in_group')[['token_decoded', 'score', 'baseline_rank']].to_dict(orient='records'),
        'bottom': tb[tb['group'] == 'bottom'].sort_values('rank_in_group')[['token_decoded', 'score', 'baseline_rank']].to_dict(orient='records'),
        'baselineRefKey': baseline_ref_key(row['model_nickname'], TEMPLATE_TO_BASELINE_COL[row['template']]),
    }

CATEGORY_ORDER = ['race', 'gender', 'intersectional', 'age', 'disability', 'control', 'religion', 'political']
CATEGORY_COLORS = {
    'race':           {'light': '#2a78d6', 'dark': '#3987e5'},
    'gender':         {'light': '#1baf7a', 'dark': '#199e70'},
    'intersectional': {'light': '#eda100', 'dark': '#c98500'},
    'age':            {'light': '#008300', 'dark': '#008300'},
    'disability':     {'light': '#4a3aa7', 'dark': '#9085e9'},
    'control':        {'light': '#e34948', 'dark': '#e66767'},
    'religion':       {'light': '#e87ba4', 'dark': '#d55181'},
    'political':      {'light': '#eb6834', 'dark': '#d95926'},
}
present_categories = [c for c in CATEGORY_ORDER if c in df['persona_category'].unique()]

records = df.to_dict(orient='records')
for r in records:
    r['key'] = detail_key(r['model_nickname'], r['template'], r['persona'])

with open(ROOT / 'config' / 'personas.yaml') as f:
    n_total_personas = len(yaml.safe_load(f)['personas'])
n_done = len(df)
vocab_sizes = df[['model_nickname', 'n_tokens']].drop_duplicates().sort_values('model_nickname')
vocab_size_text = ' &middot; '.join(
    f"{row.model_nickname}: {row.n_tokens:,} tokens" for row in vocab_sizes.itertuples()
)

# --- Prompt-framing comparison: aggregate template-pair correlations (mean across
# personas) in Python so the JS only ever renders precomputed numbers. ---
if not template_pair_df.empty:
    pair_agg = (template_pair_df
        .groupby(['model_nickname', 'question_group', 'template_a', 'template_b'])
        .agg(mean_tau=('kendall_tau', 'mean'),
             mean_rho=('spearman_rho', 'mean'),
             mean_rbo95=('rbo_p0.95', 'mean'),
             n_personas=('persona', 'count'))
        .reset_index())
else:
    pair_agg = pd.DataFrame(columns=['model_nickname', 'question_group', 'template_a', 'template_b',
                                      'mean_tau', 'mean_rho', 'mean_rbo95', 'n_personas'])

# Callout explicitly comparing the "I/you/people think" framings against the plain
# "what is" framing, per model, since that's the specific comparison asked for.
framing_callouts = {}
for nickname in pair_agg['model_nickname'].unique():
    sub = pair_agg[pair_agg['model_nickname'] == nickname]

    def _tau_for(a, b, sub=sub):
        row = sub[((sub.template_a == a) & (sub.template_b == b)) |
                   ((sub.template_a == b) & (sub.template_b == a))]
        return None if row.empty else round(float(row.iloc[0].mean_tau), 3)

    parts = []
    for other, label in [('greatest_self_id_think', 'what do <em>I</em> think'),
                          ('greatest_self_id_you_think', 'what do <em>you</em> think'),
                          ('greatest_self_id_people_think', 'what do <em>people</em> think')]:
        tau = _tau_for('greatest_self_id', other)
        if tau is not None:
            parts.append(f"&ldquo;{label} is the greatest thing ever?&rdquo; vs. plain "
                          f"&ldquo;what is&rdquo;: mean &tau; = {tau}")
    framing_callouts[nickname] = (
        '<ul>' + ''.join(f'<li>{p}</li>' for p in parts) + '</ul>'
    ) if parts else '<p class="hint">no data yet</p>'

# --- Token-level drilldown for a clicked heatmap cell (one template pair) ---
# pair_personas: which personas have data for a given (model, question_group, template_a,
# template_b) pair, for the persona picker in the drilldown panel.
def pair_key(model_nickname, question_group, template_a, template_b):
    return f"{model_nickname}|{question_group}|{template_a}|{template_b}"


pair_personas = {}
if not template_pair_df.empty:
    for (mn, qg, ta, tb), group in template_pair_df.groupby(
            ['model_nickname', 'question_group', 'template_a', 'template_b']):
        pair_personas[pair_key(mn, qg, ta, tb)] = sorted(group['persona'].unique().tolist())

pair_details = {}
if not template_pair_rank_shift_df.empty:
    for (mn, qg, ta, tb, persona), group in template_pair_rank_shift_df.groupby(
            ['model_nickname', 'question_group', 'template_a', 'template_b', 'persona']):
        key = pair_key(mn, qg, ta, tb) + f"|{persona}"
        up = group[group['direction'] == 'up'].sort_values('rank_diff').head(10)
        down = group[group['direction'] == 'down'].sort_values('rank_diff', ascending=False).head(10)
        pair_details[key] = {
            'template_a': ta,
            'template_b': tb,
            'up': up[['token_decoded', 'template_a_rank', 'template_b_rank', 'rank_diff']].to_dict(orient='records'),
            'down': down[['token_decoded', 'template_a_rank', 'template_b_rank', 'rank_diff']].to_dict(orient='records'),
        }

data_json = json.dumps(records)
categories_json = json.dumps(present_categories)
colors_json = json.dumps(CATEGORY_COLORS)
details_json = json.dumps(details)
baseline_refs_json = json.dumps(baseline_refs)
# Union with template_pair_df's models too — the framing panel has its own,
# baseline-independent data source, so it should stay usable even when the main
# summary_correlations.csv is empty (e.g. no baseline data yet for any active template).
all_model_nicknames = set(df['model_nickname'].unique()) | set(template_pair_df['model_nickname'].unique())
models_json = json.dumps(sorted(all_model_nicknames))
template_to_baseline_json = json.dumps(TEMPLATE_TO_BASELINE_COL)
template_texts_json = json.dumps(TEMPLATE_TEXTS)
pair_agg_json = json.dumps(pair_agg.to_dict(orient='records'))
framing_callouts_json = json.dumps(framing_callouts)
pair_personas_json = json.dumps(pair_personas)
pair_details_json = json.dumps(pair_details)

html = f"""<title>Persona-Conditioned Reward Model Rankings</title>
<style>
  .viz-root {{
    --surface-1:      #fcfcfb;
    --surface-2:      #f3f2ef;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #83817a;
    --border:         #e2e0da;
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      --surface-1:      #1a1a19;
      --surface-2:      #242422;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #8f8d84;
      --border:         #35342f;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    --surface-1:      #1a1a19;
    --surface-2:      #242422;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #8f8d84;
    --border:         #35342f;
  }}
  :root[data-theme="light"] .viz-root {{
    --surface-1:      #fcfcfb;
    --surface-2:      #f3f2ef;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #83817a;
    --border:         #e2e0da;
  }}

  * {{ box-sizing: border-box; }}
  body {{ margin: 0; }}
  .viz-root {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--surface-1);
    color: var(--text-primary);
    padding: 32px 24px 64px;
    min-height: 100vh;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 0.92rem; margin: 0 0 4px; }}
  .progress {{ color: var(--text-muted); font-size: 0.82rem; margin: 0 0 28px; }}
  .panel {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px 24px;
    margin-bottom: 28px;
    overflow-x: auto;
  }}
  .panel h2 {{ font-size: 1.02rem; margin: 0 0 2px; }}
  .panel .desc {{ color: var(--text-secondary); font-size: 0.85rem; margin: 0 0 16px; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 0 0 14px; font-size: 0.82rem; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 3px; flex: none; }}
  svg text {{ fill: var(--text-secondary); font-size: 11.5px; }}
  svg .axis-line {{ stroke: var(--border); stroke-width: 1; }}
  svg .zero-line {{ stroke: var(--text-muted); stroke-width: 1; }}
  .bar {{ cursor: pointer; }}
  .bar:hover, .dot:hover {{ opacity: 0.82; }}
  .dot {{ cursor: pointer; stroke: var(--surface-2); stroke-width: 2px; }}
  .tooltip {{
    position: fixed;
    pointer-events: none;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 11px;
    font-size: 0.8rem;
    color: var(--text-primary);
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    opacity: 0;
    transition: opacity 0.08s;
    z-index: 10;
    max-width: 260px;
  }}
  .tooltip .t-title {{ font-weight: 600; margin-bottom: 3px; }}
  .tooltip .t-row {{ color: var(--text-secondary); }}
  .toggle-btn {{
    font-size: 0.78rem;
    color: var(--text-secondary);
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    cursor: pointer;
    margin-bottom: 12px;
  }}
  .toggle-btn:hover {{ background: var(--surface-1); }}
  .controls {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
    margin: 0 0 20px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    position: sticky;
    top: 10px;
    z-index: 20;
  }}
  .controls label {{ display: flex; align-items: center; gap: 8px; }}
  .controls select {{
    font: inherit;
    color: var(--text-primary);
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
  }}
  .heatmap-wrap {{ margin-bottom: 20px; }}
  .heatmap-title {{ font-size: 0.85rem; font-weight: 600; margin: 0 0 8px; }}
  table.heatmap {{ border-collapse: separate; border-spacing: 2px; width: auto; table-layout: fixed; }}
  table.heatmap th, table.heatmap td {{ border-bottom: none; text-align: center; white-space: normal; }}
  table.heatmap th {{
    font-weight: 600;
    font-size: 0.74rem;
    line-height: 1.25;
    padding: 6px 6px;
    width: 108px;
    word-break: break-word;
    overflow-wrap: anywhere;
    hyphens: auto;
  }}
  table.heatmap thead th:first-child, table.heatmap tbody th {{ width: 150px; text-align: right; }}
  table.heatmap td {{ padding: 10px 8px; border-radius: 4px; font-variant-numeric: tabular-nums; cursor: default; }}
  .subpanel {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px 16px;
    margin-top: 16px;
  }}
  .subpanel label {{ font-size: 0.85rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; }}
  .callout {{ margin: 0 0 18px; }}
  .callout ul {{ margin: 0; padding-left: 20px; color: var(--text-secondary); font-size: 0.85rem; }}
  .callout li {{ margin-bottom: 4px; }}
  code {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; font-size: 0.85em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ color: var(--text-secondary); font-weight: 600; }}
  td {{ color: var(--text-primary); }}
  .note {{ color: var(--text-muted); font-size: 0.78rem; margin-top: 10px; }}
  .hint {{ color: var(--text-muted); font-size: 0.82rem; font-style: italic; }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 640px) {{ .detail-grid {{ grid-template-columns: 1fr; }} }}
  .detail-col h3 {{ font-size: 0.85rem; margin: 0 0 8px; color: var(--text-primary); }}
  .detail-col table {{ font-size: 0.78rem; }}
  .up-val {{ color: #1baf7a; }}
  .down-val {{ color: #e34948; }}
  @media (prefers-color-scheme: dark) {{
    .up-val {{ color: #199e70; }}
    .down-val {{ color: #e66767; }}
  }}
  .selected-label {{ font-weight: 600; margin-bottom: 14px; font-size: 0.95rem; }}
</style>

<div class="viz-root">
  <div class="wrap">
    <h1>Persona-Conditioned Reward Model Rankings</h1>
    <p class="subtitle" id="subtitle"></p>
    <p class="progress">{n_done} of {n_total_personas} personas scored so far &middot; live run in progress</p>
    <p class="progress">Full vocabulary scored per prompt (every token, exhaustively) &mdash; {vocab_size_text}</p>

    <div class="controls">
      <label>Model <select id="model-select"></select></label>
      <label>Template <select id="template-select"></select></label>
    </div>

    <div class="panel">
      <h2>Reference: the neutral prompt, with no persona at all</h2>
      <p class="desc">What the model prefers with no "I am X" prefix &mdash; e.g. &ldquo;What, in one word, is the greatest thing ever?&rdquo;. Every persona comparison below is measured against this.</p>
      <div id="baseline-body"></div>
    </div>

    <div class="panel">
      <h2>Agreement with the neutral baseline, by persona</h2>
      <p class="desc">Kendall&rsquo;s &tau; across the full token vocabulary. Bars near zero mean the persona-conditioned ranking has little in common with the plain, no-persona prompt. Solid bars are the reward model; lighter dashed bars (when present) are the base LM's own persona-shift &tau; for comparison. <span class="hint">Click a bar for token-level detail.</span></p>
      <div class="legend" id="legend-bar"></div>
      <svg id="bar-chart"></svg>
    </div>

    <div class="panel" id="detail-panel">
      <h2>Token-level detail</h2>
      <p class="desc" id="detail-hint">Click a persona above to see its highest/lowest scoring tokens and the tokens whose rank shifted most vs. the neutral baseline.</p>
      <div id="detail-body"></div>
    </div>

    <div class="panel">
      <h2>Whole-distribution vs. top-of-ranking agreement</h2>
      <p class="desc">&tau; captures agreement across the whole vocabulary; RBO (p=0.95) weights the very top of the ranking &mdash; i.e. what the model would actually prefer. Personas low on RBO but not-as-low on &tau; are reshaping the top answer specifically, not the whole distribution. <span class="hint">Click a dot for token-level detail.</span></p>
      <div class="legend" id="legend-scatter"></div>
      <svg id="scatter-chart"></svg>
    </div>

    <div class="panel" id="lm-divergence-panel">
      <h2>Base LM vs. reward model persona sensitivity</h2>
      <p class="desc">x = the base LM's own persona-shift &tau; (its persona-conditioned logits vs. its own neutral baseline); y = the reward model's persona-shift &tau; for the same persona/template. Points near the diagonal mean the RM mostly inherits the base LM's persona sensitivity; points off the diagonal mean reward-model finetuning changed it.</p>
      <svg id="lm-divergence-chart"></svg>
      <p class="hint" id="lm-divergence-hint" style="display:none;">No base-LM logit data yet for this model &mdash; run <code>generate_base_model_logprobs.py --config llama_base_models.yaml</code> and <code>generate_persona_base_model_logprobs.py</code>, then rerun <code>analyze_personas.py</code>.</p>
    </div>

    <div class="panel" id="framing-panel">
      <h2>Prompt-framing comparison</h2>
      <p class="desc">How similar are different phrasings of &ldquo;the same&rdquo; question, for the same persona? Mean Kendall&rsquo;s &tau; across all scored personas, between every pair of templates that share an underlying question (e.g. the four &ldquo;greatest thing ever&rdquo; framings). One heatmap per question group. <span class="hint">Click a cell for token-level detail.</span></p>
      <div class="callout" id="framing-callout"></div>
      <div id="framing-heatmaps"></div>
      <div class="subpanel" id="pair-drilldown-panel" style="display:none;"></div>
    </div>

    <button class="toggle-btn" id="table-toggle">Show data table</button>
    <div class="panel" id="table-panel" style="display:none;"></div>

    <p class="note">Neutral baseline: reward model score ranking for &ldquo;What, in one word, is the greatest thing ever?&rdquo; with no persona prefix (from the original paper&rsquo;s data). Persona categories from Wang et al. (2025).</p>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const data = {data_json};
const categories = {categories_json};
const colors = {colors_json};
const details = {details_json};
const baselineRefs = {baseline_refs_json};
const models = {models_json};
const templateToBaseline = {template_to_baseline_json};
const templateTexts = {template_texts_json};
const pairAgg = {pair_agg_json};
const framingCallouts = {framing_callouts_json};
const pairPersonas = {pair_personas_json};
const pairDetails = {pair_details_json};
let selectedKey = null;
let selectedModel = models[0] || null;
let selectedTemplate = null;

function templatesForModel(modelNickname) {{
  return [...new Set(data.filter(d => d.model_nickname === modelNickname).map(d => d.template))].sort();
}}
function populateSelect(sel, options) {{
  sel.innerHTML = options.map(o => `<option value="${{o}}">${{o}}</option>`).join('');
}}
function currentRows() {{
  return data.filter(d => d.model_nickname === selectedModel && d.template === selectedTemplate);
}}

function isDark() {{
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}}

function colorFor(cat) {{
  const mode = isDark() ? 'dark' : 'light';
  return (colors[cat] || {{light:'#888',dark:'#aaa'}})[mode];
}}

// Diverging blue<->red scale (reference dataviz palette), gray at the midpoint,
// for the tau-valued (-1..1) prompt-framing heatmap cells.
function lerpHex(hexA, hexB, t) {{
  const a = [1,3,5].map(i => parseInt(hexA.slice(i, i + 2), 16));
  const b = [1,3,5].map(i => parseInt(hexB.slice(i, i + 2), 16));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
}}
function divergingColor(t) {{
  const mode = isDark() ? 'dark' : 'light';
  const red = mode === 'dark' ? '#e66767' : '#e34948';
  const blue = mode === 'dark' ? '#3987e5' : '#2a78d6';
  const mid = mode === 'dark' ? '#383835' : '#f0efec';
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? lerpHex(mid, red, -t) : lerpHex(mid, blue, t);
}}

const tooltip = document.getElementById('tooltip');
function showTooltip(evt, html) {{
  tooltip.innerHTML = html;
  tooltip.style.opacity = 1;
  const pad = 14;
  tooltip.style.left = (evt.clientX + pad) + 'px';
  tooltip.style.top = (evt.clientY + pad) + 'px';
}}
function hideTooltip() {{ tooltip.style.opacity = 0; }}

function buildLegend(container) {{
  container.innerHTML = '';
  categories.forEach(cat => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = colorFor(cat);
    item.appendChild(sw);
    const label = document.createElement('span');
    label.textContent = cat;
    item.appendChild(label);
    container.appendChild(item);
  }});
}}
buildLegend(document.getElementById('legend-bar'));
buildLegend(document.getElementById('legend-scatter'));

// ---------------- Bar chart: Kendall's tau by persona ----------------
function drawBarChart() {{
  const svg = document.getElementById('bar-chart');
  const sorted = [...currentRows()].sort((a, b) => a.kendall_tau - b.kendall_tau);
  const barH = 24, gap = 10, leftPad = 150, rightPad = 40, topPad = 8;
  const rowH = barH + gap;
  const width = 760;
  const plotW = width - leftPad - rightPad;
  const height = topPad + sorted.length * rowH + 30;
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = '';

  const xMax = 1.0, xMin = -1.0;
  const xScale = v => leftPad + ((v - xMin) / (xMax - xMin)) * plotW;
  const zeroX = xScale(0);

  // axis ticks
  [-1, -0.5, 0, 0.5, 1].forEach(t => {{
    const x = xScale(t);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x); line.setAttribute('x2', x);
    line.setAttribute('y1', topPad); line.setAttribute('y2', topPad + sorted.length * rowH);
    line.setAttribute('class', t === 0 ? 'zero-line' : 'axis-line');
    svg.appendChild(line);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', x); label.setAttribute('y', topPad + sorted.length * rowH + 18);
    label.setAttribute('text-anchor', 'middle');
    label.textContent = t.toFixed(1);
    svg.appendChild(label);
  }});

  sorted.forEach((d, i) => {{
    const y = topPad + i * rowH;
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', leftPad - 10); label.setAttribute('y', y + barH / 2 + 4);
    label.setAttribute('text-anchor', 'end');
    label.textContent = d.persona;
    svg.appendChild(label);

    const rmBarH = d.lm_available ? barH / 2 - 1 : barH;
    const x0 = Math.min(zeroX, xScale(d.kendall_tau));
    const w = Math.abs(xScale(d.kendall_tau) - zeroX);
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x0); rect.setAttribute('y', y);
    rect.setAttribute('width', Math.max(w, 1)); rect.setAttribute('height', rmBarH);
    rect.setAttribute('rx', 4);
    rect.setAttribute('fill', colorFor(d.persona_category));
    rect.setAttribute('class', 'bar');
    if (d.key === selectedKey) {{
      rect.setAttribute('stroke', 'var(--text-primary)');
      rect.setAttribute('stroke-width', '2');
    }}
    rect.addEventListener('mousemove', e => showTooltip(e,
      `<div class="t-title">${{d.persona}}</div>` +
      `<div class="t-row">category: ${{d.persona_category}}</div>` +
      `<div class="t-row">Kendall &tau;: ${{d.kendall_tau.toFixed(3)}}</div>` +
      `<div class="t-row">Spearman &rho;: ${{d.spearman_rho.toFixed(3)}}</div>` +
      `<div class="t-row">RBO (p=0.95): ${{d['rbo_p0.95'].toFixed(3)}}</div>`));
    rect.addEventListener('mouseleave', hideTooltip);
    rect.addEventListener('click', () => selectPersona(d));
    svg.appendChild(rect);

    if (d.lm_available) {{
      const lmX0 = Math.min(zeroX, xScale(d.lm_kendall_tau));
      const lmW = Math.abs(xScale(d.lm_kendall_tau) - zeroX);
      const lmRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      lmRect.setAttribute('x', lmX0); lmRect.setAttribute('y', y + barH / 2 + 1);
      lmRect.setAttribute('width', Math.max(lmW, 1)); lmRect.setAttribute('height', barH / 2 - 1);
      lmRect.setAttribute('rx', 3);
      lmRect.setAttribute('fill', colorFor(d.persona_category));
      lmRect.setAttribute('fill-opacity', '0.45');
      lmRect.setAttribute('stroke-dasharray', '3,2');
      lmRect.setAttribute('class', 'bar');
      lmRect.addEventListener('mousemove', e => showTooltip(e,
        `<div class="t-title">${{d.persona}} &mdash; base LM</div>` +
        `<div class="t-row">Base LM Kendall &tau;: ${{d.lm_kendall_tau.toFixed(3)}}</div>` +
        `<div class="t-row">Reward model Kendall &tau;: ${{d.kendall_tau.toFixed(3)}}</div>`));
      lmRect.addEventListener('mouseleave', hideTooltip);
      lmRect.addEventListener('click', () => selectPersona(d));
      svg.appendChild(lmRect);
    }}
  }});
}}

// ---------------- Scatter: tau vs RBO(0.95) ----------------
function drawScatter() {{
  const svg = document.getElementById('scatter-chart');
  const width = 760, height = 420;
  const pad = {{left: 56, right: 24, top: 16, bottom: 44}};
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = '';

  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xMin = 0, xMax = 1, yMin = 0, yMax = 1;
  const xScale = v => pad.left + (v - xMin) / (xMax - xMin) * plotW;
  const yScale = v => pad.top + plotH - (v - yMin) / (yMax - yMin) * plotH;

  [0, 0.25, 0.5, 0.75, 1].forEach(t => {{
    const x = xScale(t), y = yScale(t);
    const vline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    vline.setAttribute('x1', x); vline.setAttribute('x2', x);
    vline.setAttribute('y1', pad.top); vline.setAttribute('y2', pad.top + plotH);
    vline.setAttribute('class', 'axis-line');
    svg.appendChild(vline);
    const hline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hline.setAttribute('x1', pad.left); hline.setAttribute('x2', pad.left + plotW);
    hline.setAttribute('y1', y); hline.setAttribute('y2', y);
    hline.setAttribute('class', 'axis-line');
    svg.appendChild(hline);
    const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xl.setAttribute('x', x); xl.setAttribute('y', pad.top + plotH + 18);
    xl.setAttribute('text-anchor', 'middle'); xl.textContent = t.toFixed(2);
    svg.appendChild(xl);
    const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yl.setAttribute('x', pad.left - 8); yl.setAttribute('y', y + 4);
    yl.setAttribute('text-anchor', 'end'); yl.textContent = t.toFixed(2);
    svg.appendChild(yl);
  }});

  const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  xLabel.setAttribute('x', pad.left + plotW / 2); xLabel.setAttribute('y', height - 4);
  xLabel.setAttribute('text-anchor', 'middle');
  xLabel.textContent = "Kendall's tau (whole distribution)";
  svg.appendChild(xLabel);

  const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  yLabel.setAttribute('x', -(pad.top + plotH / 2)); yLabel.setAttribute('y', 14);
  yLabel.setAttribute('text-anchor', 'middle');
  yLabel.setAttribute('transform', 'rotate(-90)');
  yLabel.textContent = 'RBO, p=0.95 (top-ranks agreement)';
  svg.appendChild(yLabel);

  currentRows().forEach(d => {{
    const cx = xScale(d.kendall_tau), cy = yScale(d['rbo_p0.95']);
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('r', 6);
    dot.setAttribute('fill', colorFor(d.persona_category));
    dot.setAttribute('class', 'dot');
    if (d.key === selectedKey) {{
      dot.setAttribute('r', 8);
      dot.setAttribute('stroke', 'var(--text-primary)');
    }}
    dot.addEventListener('mousemove', e => showTooltip(e,
      `<div class="t-title">${{d.persona}}</div>` +
      `<div class="t-row">category: ${{d.persona_category}}</div>` +
      `<div class="t-row">Kendall &tau;: ${{d.kendall_tau.toFixed(3)}}</div>` +
      `<div class="t-row">RBO (p=0.95): ${{d['rbo_p0.95'].toFixed(3)}}</div>`));
    dot.addEventListener('mouseleave', hideTooltip);
    dot.addEventListener('click', () => selectPersona(d));
    svg.appendChild(dot);
  }});
}}

// ---------------- RM-vs-LM divergence scatter ----------------
function drawLmDivergenceScatter() {{
  const svg = document.getElementById('lm-divergence-chart');
  const hint = document.getElementById('lm-divergence-hint');
  const rows = currentRows().filter(d => d.lm_available);

  if (rows.length === 0) {{
    svg.style.display = 'none';
    hint.style.display = 'block';
    return;
  }}
  svg.style.display = '';
  hint.style.display = 'none';

  const width = 760, height = 420;
  const pad = {{left: 56, right: 24, top: 16, bottom: 44}};
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = '';

  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xMin = -1, xMax = 1, yMin = -1, yMax = 1;
  const xScale = v => pad.left + (v - xMin) / (xMax - xMin) * plotW;
  const yScale = v => pad.top + plotH - (v - yMin) / (yMax - yMin) * plotH;

  [-1, -0.5, 0, 0.5, 1].forEach(t => {{
    const x = xScale(t), y = yScale(t);
    const vline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    vline.setAttribute('x1', x); vline.setAttribute('x2', x);
    vline.setAttribute('y1', pad.top); vline.setAttribute('y2', pad.top + plotH);
    vline.setAttribute('class', t === 0 ? 'zero-line' : 'axis-line');
    svg.appendChild(vline);
    const hline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hline.setAttribute('x1', pad.left); hline.setAttribute('x2', pad.left + plotW);
    hline.setAttribute('y1', y); hline.setAttribute('y2', y);
    hline.setAttribute('class', t === 0 ? 'zero-line' : 'axis-line');
    svg.appendChild(hline);
    const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xl.setAttribute('x', x); xl.setAttribute('y', pad.top + plotH + 18);
    xl.setAttribute('text-anchor', 'middle'); xl.textContent = t.toFixed(1);
    svg.appendChild(xl);
    const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yl.setAttribute('x', pad.left - 8); yl.setAttribute('y', y + 4);
    yl.setAttribute('text-anchor', 'end'); yl.textContent = t.toFixed(1);
    svg.appendChild(yl);
  }});

  const diag = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  diag.setAttribute('x1', xScale(xMin)); diag.setAttribute('y1', yScale(yMin));
  diag.setAttribute('x2', xScale(xMax)); diag.setAttribute('y2', yScale(yMax));
  diag.setAttribute('class', 'axis-line');
  diag.setAttribute('stroke-dasharray', '4,3');
  svg.appendChild(diag);

  const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  xLabel.setAttribute('x', pad.left + plotW / 2); xLabel.setAttribute('y', height - 4);
  xLabel.setAttribute('text-anchor', 'middle');
  xLabel.textContent = "Base LM's own persona-shift Kendall's tau";
  svg.appendChild(xLabel);

  const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  yLabel.setAttribute('x', -(pad.top + plotH / 2)); yLabel.setAttribute('y', 14);
  yLabel.setAttribute('text-anchor', 'middle');
  yLabel.setAttribute('transform', 'rotate(-90)');
  yLabel.textContent = "Reward model's persona-shift Kendall's tau";
  svg.appendChild(yLabel);

  rows.forEach(d => {{
    const cx = xScale(d.lm_kendall_tau), cy = yScale(d.kendall_tau);
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('r', 6);
    dot.setAttribute('fill', colorFor(d.persona_category));
    dot.setAttribute('class', 'dot');
    if (d.key === selectedKey) {{
      dot.setAttribute('r', 8);
      dot.setAttribute('stroke', 'var(--text-primary)');
    }}
    dot.addEventListener('mousemove', e => showTooltip(e,
      `<div class="t-title">${{d.persona}}</div>` +
      `<div class="t-row">category: ${{d.persona_category}}</div>` +
      `<div class="t-row">Base LM &tau;: ${{d.lm_kendall_tau.toFixed(3)}}</div>` +
      `<div class="t-row">Reward model &tau;: ${{d.kendall_tau.toFixed(3)}}</div>`));
    dot.addEventListener('mouseleave', hideTooltip);
    dot.addEventListener('click', () => selectPersona(d));
    svg.appendChild(dot);
  }});
}}

// ---------------- Detail panel: token-level drilldown for one persona ----------------
function tokenTable(rows, valueLabel, valueFn, showBaselineRank) {{
  if (!rows || rows.length === 0) return '<p class="hint">no data yet</p>';
  const extraHeader = showBaselineRank ? '<th>rank under neutral prompt</th>' : '';
  const body = rows.map(r => {{
    const baselineCell = showBaselineRank
      ? `<td class="hint">#${{Math.round(r.baseline_rank).toLocaleString()}}</td>` : '';
    return `<tr><td>${{r.token_decoded}}</td><td>${{valueFn(r)}}</td>${{baselineCell}}</tr>`;
  }}).join('');
  return `<table><thead><tr><th>token</th><th>${{valueLabel}}</th>${{extraHeader}}</tr></thead><tbody>${{body}}</tbody></table>`;
}}

// ---------------- Baseline reference panel (no persona at all) ----------------
function renderBaselinePanel(modelNickname, baselineCol) {{
  const container = document.getElementById('baseline-body');
  const key = `${{modelNickname}}|${{baselineCol}}`;
  const ref = baselineRefs[key];
  if (!ref) {{
    container.innerHTML = '<p class="hint">no data yet</p>';
    return;
  }}
  container.innerHTML = `
    <p class="selected-label">${{modelNickname}} &mdash; &ldquo;What is the ${{baselineCol}} thing ever?&rdquo; (no persona)</p>
    <div class="detail-grid" style="margin-bottom:20px;">
      <div class="detail-col">
        <h3>Highest-scoring tokens</h3>
        ${{tokenTable(ref.top, 'score', r => r.score.toFixed(2))}}
      </div>
      <div class="detail-col">
        <h3>Lowest-scoring tokens</h3>
        ${{tokenTable(ref.bottom, 'score', r => r.score.toFixed(2))}}
      </div>
    </div>`;
}}

// ---------------- Prompt-framing comparison panel ----------------
function prettyTemplateLabel(t) {{ return t.replace(/_/g, ' '); }}

function renderOneHeatmap(questionGroup, templateNames) {{
  const lookup = {{}};
  pairAgg.filter(r => r.model_nickname === selectedModel && r.question_group === questionGroup)
    .forEach(r => {{
      lookup[`${{r.template_a}}|${{r.template_b}}`] = r;
      lookup[`${{r.template_b}}|${{r.template_a}}`] = r;
    }});

  const rowsHtml = templateNames.map(rowT => {{
    const cells = templateNames.map(colT => {{
      if (rowT === colT) {{
        return `<td style="background:${{divergingColor(1)}};color:#fff;">1.00</td>`;
      }}
      const r = lookup[`${{rowT}}|${{colT}}`];
      if (!r) return `<td class="hint">&mdash;</td>`;
      const bg = divergingColor(r.mean_tau);
      const textColor = Math.abs(r.mean_tau) > 0.4 ? '#fff' : 'var(--text-primary)';
      return `<td style="background:${{bg}};color:${{textColor}};cursor:pointer;" ` +
        `data-tooltip-tau="${{r.mean_tau.toFixed(3)}}" data-tooltip-rho="${{r.mean_rho.toFixed(3)}}" ` +
        `data-tooltip-rbo="${{r.mean_rbo95.toFixed(3)}}" data-tooltip-n="${{r.n_personas}}" ` +
        `data-row-t="${{rowT}}" data-col-t="${{colT}}">${{r.mean_tau.toFixed(2)}}</td>`;
    }}).join('');
    return `<tr><th>${{prettyTemplateLabel(rowT)}}</th>${{cells}}</tr>`;
  }}).join('');

  return `
    <div class="heatmap-wrap">
      <p class="heatmap-title">&ldquo;${{questionGroup}}&rdquo; framings &mdash; mean Kendall &tau; across personas</p>
      <div style="overflow-x:auto;">
        <table class="heatmap" id="heatmap-${{questionGroup}}">
          <thead><tr><th></th>${{templateNames.map(t => `<th>${{prettyTemplateLabel(t)}}</th>`).join('')}}</tr></thead>
          <tbody>${{rowsHtml}}</tbody>
        </table>
      </div>
    </div>`;
}}

function renderFramingPanel() {{
  document.getElementById('framing-callout').innerHTML =
    framingCallouts[selectedModel] || '<p class="hint">no data yet</p>';

  const groups = {{}};
  pairAgg.filter(r => r.model_nickname === selectedModel).forEach(r => {{
    groups[r.question_group] = groups[r.question_group] || new Set();
    groups[r.question_group].add(r.template_a);
    groups[r.question_group].add(r.template_b);
  }});
  const container = document.getElementById('framing-heatmaps');
  const groupKeys = Object.keys(groups).filter(k => groups[k].size >= 2);
  if (groupKeys.length === 0) {{
    container.innerHTML = '<p class="hint">no data yet</p>';
    return;
  }}
  container.innerHTML = groupKeys.map(bc => renderOneHeatmap(bc, [...groups[bc]].sort())).join('');

  groupKeys.forEach(bc => {{
    document.querySelectorAll(`#heatmap-${{bc}} td[data-tooltip-tau]`).forEach(td => {{
      td.addEventListener('mousemove', e => showTooltip(e,
        `<div class="t-row">Kendall &tau;: ${{td.dataset.tooltipTau}}</div>` +
        `<div class="t-row">Spearman &rho;: ${{td.dataset.tooltipRho}}</div>` +
        `<div class="t-row">RBO (p=0.95): ${{td.dataset.tooltipRbo}}</div>` +
        `<div class="t-row">averaged over ${{td.dataset.tooltipN}} personas</div>`));
      td.addEventListener('mouseleave', hideTooltip);
      td.addEventListener('click', () => showPairDrilldown(bc, td.dataset.rowT, td.dataset.colT));
    }});
  }});

  // Previously-shown drilldown may no longer apply after a model/template switch —
  // hide it until the user clicks a cell again.
  document.getElementById('pair-drilldown-panel').style.display = 'none';
}}

// ---------------- Token-level drilldown for a clicked heatmap cell ----------------
function pairPersonasFor(questionGroup, templateA, templateB) {{
  return pairPersonas[`${{selectedModel}}|${{questionGroup}}|${{templateA}}|${{templateB}}`] ||
         pairPersonas[`${{selectedModel}}|${{questionGroup}}|${{templateB}}|${{templateA}}`] || [];
}}

function pairTokenTable(rows, labelA, labelB, colorClass, sign) {{
  if (!rows || rows.length === 0) return '<p class="hint">no data yet</p>';
  const body = rows.map(r => `<tr><td>${{r.token_decoded}}</td>` +
    `<td class="hint">#${{Math.round(r.template_a_rank).toLocaleString()}}</td>` +
    `<td class="hint">#${{Math.round(r.template_b_rank).toLocaleString()}}</td>` +
    `<td><span class="${{colorClass}}">${{sign}}${{Math.abs(Math.round(r.rank_diff)).toLocaleString()}}</span></td></tr>`).join('');
  return `<table><thead><tr><th>token</th><th>rank (${{prettyTemplateLabel(labelA)}})</th>` +
    `<th>rank (${{prettyTemplateLabel(labelB)}})</th><th>rank shift</th></tr></thead><tbody>${{body}}</tbody></table>`;
}}

function renderPairDetail(questionGroup, templateA, templateB, persona) {{
  const panel = document.getElementById('pair-drilldown-panel');
  const personas = pairPersonasFor(questionGroup, templateA, templateB);
  const personaOptions = personas.map(p =>
    `<option value="${{p}}" ${{p === persona ? 'selected' : ''}}>${{p}}</option>`).join('');

  const key = `${{selectedModel}}|${{questionGroup}}|${{templateA}}|${{templateB}}|${{persona}}`;
  const revKey = `${{selectedModel}}|${{questionGroup}}|${{templateB}}|${{templateA}}|${{persona}}`;
  const det = pairDetails[key] || pairDetails[revKey];

  const header = `
    <p class="selected-label">${{prettyTemplateLabel(templateA)}} vs. ${{prettyTemplateLabel(templateB)}} &mdash; token rank shifts</p>
    <label>Persona <select id="pair-persona-select">${{personaOptions}}</select></label>`;

  if (!det) {{
    panel.innerHTML = header + '<p class="hint" style="margin-top:10px;">no token-level detail yet for this persona/pair</p>';
  }} else {{
    const labelA = det.template_a, labelB = det.template_b;
    panel.innerHTML = header + `
      <div class="detail-grid" style="margin-top:12px;">
        <div class="detail-col">
          <h3>Moved to a <span class="up-val">much better</span> rank under &ldquo;${{prettyTemplateLabel(labelB)}}&rdquo;</h3>
          ${{pairTokenTable(det.up, labelA, labelB, 'up-val', '-')}}
        </div>
        <div class="detail-col">
          <h3>Moved to a <span class="down-val">much worse</span> rank under &ldquo;${{prettyTemplateLabel(labelB)}}&rdquo;</h3>
          ${{pairTokenTable(det.down, labelA, labelB, 'down-val', '+')}}
        </div>
      </div>`;
  }}
  panel.style.display = 'block';
  const sel = document.getElementById('pair-persona-select');
  if (sel) sel.addEventListener('change', e => renderPairDetail(questionGroup, templateA, templateB, e.target.value));
}}

function showPairDrilldown(questionGroup, templateA, templateB) {{
  const personas = pairPersonasFor(questionGroup, templateA, templateB);
  if (personas.length === 0) return;
  renderPairDetail(questionGroup, templateA, templateB, personas[0]);
}}

function renderDetail(d) {{
  selectedKey = d.key;
  document.getElementById('detail-hint').textContent =
    `"${{d.persona}}" (${{d.persona_category}}) vs. neutral baseline, ${{d.model_nickname}} / ${{d.template}}`;
  const det = details[d.key] || {{}};
  const body = document.getElementById('detail-body');
  body.innerHTML = `
    <p class="hint">"rank under neutral prompt" shows where that same token sat with no persona at all &mdash; compare against the reference panel above.</p>
    <div class="detail-grid">
      <div class="detail-col">
        <h3>Highest-scoring tokens under this persona</h3>
        ${{tokenTable(det.top, 'score', r => r.score.toFixed(2), true)}}
      </div>
      <div class="detail-col">
        <h3>Lowest-scoring tokens under this persona</h3>
        ${{tokenTable(det.bottom, 'score', r => r.score.toFixed(2), true)}}
      </div>
      <div class="detail-col">
        <h3>Moved to a <span class="up-val">much better</span> rank vs. baseline</h3>
        ${{tokenTable(det.up, 'rank shift', r => '<span class="up-val">' + Math.round(r.rank_diff).toLocaleString() + '</span>')}}
      </div>
      <div class="detail-col">
        <h3>Moved to a <span class="down-val">much worse</span> rank vs. baseline</h3>
        ${{tokenTable(det.down, 'rank shift', r => '<span class="down-val">+' + Math.round(r.rank_diff).toLocaleString() + '</span>')}}
      </div>
    </div>`;
  drawBarChart();
  drawScatter();
  drawLmDivergenceScatter();
}}

function selectPersona(d) {{ renderDetail(d); }}

// ---------------- Wiring: dropdowns drive every panel ----------------
function refreshAll() {{
  const modelRow = data.find(d => d.model_nickname === selectedModel);
  document.getElementById('subtitle').innerHTML =
    `${{modelRow ? modelRow.model : selectedModel}} &mdash; "${{templateTexts[selectedTemplate] || ''}}" vs. the neutral, no-prefix baseline`;
  renderBaselinePanel(selectedModel, templateToBaseline[selectedTemplate]);
  drawBarChart();
  drawScatter();
  drawLmDivergenceScatter();
  renderFramingPanel();

  const inScope = currentRows();
  if (inScope.length > 0) {{
    const mostDivergent = [...inScope].sort((a, b) => a.kendall_tau - b.kendall_tau)[0];
    renderDetail(mostDivergent);
  }}
}}

const modelSelect = document.getElementById('model-select');
const templateSelect = document.getElementById('template-select');

populateSelect(modelSelect, models);
modelSelect.value = selectedModel;
selectedTemplate = templatesForModel(selectedModel)[0] || null;
populateSelect(templateSelect, templatesForModel(selectedModel));
templateSelect.value = selectedTemplate;

modelSelect.addEventListener('change', e => {{
  selectedModel = e.target.value;
  const opts = templatesForModel(selectedModel);
  populateSelect(templateSelect, opts);
  selectedTemplate = opts[0] || null;
  templateSelect.value = selectedTemplate;
  refreshAll();
}});
templateSelect.addEventListener('change', e => {{
  selectedTemplate = e.target.value;
  refreshAll();
}});

refreshAll();

// Redraw on theme change so colors stay correct
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {{
  drawBarChart(); drawScatter(); drawLmDivergenceScatter(); renderFramingPanel();
}});
new MutationObserver(() => {{
  drawBarChart(); drawScatter(); drawLmDivergenceScatter(); renderFramingPanel();
}}).observe(document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});

// ---------------- Data table toggle ----------------
document.getElementById('table-toggle').addEventListener('click', () => {{
  const panel = document.getElementById('table-panel');
  const btn = document.getElementById('table-toggle');
  if (panel.style.display === 'none') {{
    let rows = data.map(d => `<tr><td>${{d.persona}}</td><td>${{d.persona_category}}</td>` +
      `<td>${{d.kendall_tau.toFixed(3)}}</td><td>${{d.spearman_rho.toFixed(3)}}</td>` +
      `<td>${{d['rbo_p0.90'].toFixed(3)}}</td><td>${{d['rbo_p0.95'].toFixed(3)}}</td><td>${{d['rbo_p0.99'].toFixed(3)}}</td></tr>`).join('');
    panel.innerHTML = `<table><thead><tr><th>Persona</th><th>Category</th><th>Kendall &tau;</th>` +
      `<th>Spearman &rho;</th><th>RBO p=0.90</th><th>RBO p=0.95</th><th>RBO p=0.99</th></tr></thead><tbody>${{rows}}</tbody></table>`;
    panel.style.display = 'block';
    btn.textContent = 'Hide data table';
  }} else {{
    panel.style.display = 'none';
    btn.textContent = 'Show data table';
  }}
}});
</script>
"""

out_path = Path(__file__).parent / 'dashboard.html'
out_path.write_text(html)
print(f"Wrote {out_path}")
