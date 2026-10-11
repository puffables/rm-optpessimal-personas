"""
Ranking helpers shared by the persona, kv-cache, MFD and vector analyses.
"""
import re

import pandas as pd


def slugify(text):
    """Persona name -> column-name slug, e.g. 'a Black woman' -> 'a_black_woman'."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ranked_ids(df, score_col, id_col='token_id'):
    """IDs sorted best-first by score_col, with ties broken by ascending id.

    The fp16 reward scores are heavily quantised (the default-path `greatest`
    column has under 1,000 distinct values across ~128k tokens), so an
    unstable sort makes RBO and top-k lists depend on arbitrary tie order.
    Sorting by id first and then stably by score makes them deterministic.
    """
    ordered = df.sort_values(id_col, kind='mergesort')
    ordered = ordered.sort_values(score_col, ascending=False, kind='mergesort')
    return ordered[id_col].tolist()


def rbo(list1, list2, p=0.9, depth=None):
    """Rank-Biased Overlap between two ranked lists (already sorted best-first).

    Tracks the running overlap incrementally (O(n)) rather than recomputing a
    full set intersection at every depth (O(n^2)), which is intractable at
    full-vocabulary depth (~100k+ tokens). This is the truncated (lower-bound)
    RBO, without extrapolation past `depth`.
    """
    if depth is None:
        depth = min(len(list1), len(list2))
    list1, list2 = list1[:depth], list2[:depth]
    seen1, seen2 = set(), set()
    overlap = 0
    total = 0.0
    for d in range(depth):
        item1, item2 = list1[d], list2[d]
        if item1 == item2:
            overlap += 1
        else:
            if item1 in seen2:
                overlap += 1
            if item2 in seen1:
                overlap += 1
        seen1.add(item1)
        seen2.add(item2)
        total += (overlap / (d + 1)) * (p ** d)
    return (1 - p) * total


def best_per_decoded(df, sort_col, ascending=False):
    """Sort by sort_col, then keep the first row per token_decoded string.

    Many distinct token_ids decode to the same display string. Sorting before
    deduplicating keeps the best-placed variant of each string, so a top-k
    list can't lose its top token just because a lower-id duplicate exists.
    """
    ordered = df.sort_values('token_id', kind='mergesort')
    ordered = ordered.sort_values(sort_col, ascending=ascending, kind='mergesort')
    return ordered.drop_duplicates(subset='token_decoded', keep='first')


def format_token_label(token_decoded):
    """Make a decoded token safe/legible as a plot or table label.

    Distinct tokens frequently decode to strings that differ only in leading/
    trailing whitespace (e.g. BPE's "Ġ"-prefixed space marker) — real, distinct
    tokens that are kept separate everywhere upstream (joins are on token_id,
    dedup is on the exact token_decoded string), but whitespace collapses
    invisibly in both matplotlib labels and HTML, making them look like
    duplicate rows. Render whitespace visibly instead of stripping it.
    """
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
