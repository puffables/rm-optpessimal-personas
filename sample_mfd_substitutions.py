"""
Top each foundation/valence cell up to N frame-fitting nouns and print the
additions as a config/mfd_substitutions.yaml block.

The `sampled` tier is a superset of the seed tiers: each cell's 10 words are its
four MFD 2.0 seed words (already in the config, tagged `primary`/`extended`)
plus the six drawn here. So this samples N minus however many seed words that
cell already has, and the seed words keep their existing entries — nothing is
duplicated, and selecting `sampled` in the notebook yields the full 10.

Run generate_typed_dictionary.py first — this reads its output. The sample is
seeded, so re-running reproduces the same words; the printed block is pasted
into config/mfd_substitutions.yaml under tier `sampled`.

Candidates are singular entries with frame_fit == True whose word_type is
'noun' or 'gerund' — the forms that fill "I value ___ the most" without a
determiner and still agree with "___ is the greatest thing ever". Gerunds
qualify as verbal nouns, and two of the MFD 2.0 seed words (`suffering`,
`cheating`) are gerunds themselves. A word is
used at most once overall: the 63 entries filed under two foundations would
otherwise be able to represent both, and words already in the file are skipped
so no name appears twice.

    python sample_mfd_substitutions.py [--n 10] [--seed 20260813]
"""
import argparse
import collections
import random
from pathlib import Path

import pandas as pd
import yaml
from nltk.corpus import wordnet as wn

SCRIPT_ROOT = Path(__file__).parent
TYPED_DICT = SCRIPT_ROOT / 'data' / 'dictionaries' / 'mfd2.0_typed.csv'
SUBSTITUTIONS = SCRIPT_ROOT / 'config' / 'mfd_substitutions.yaml'

FOUNDATIONS = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
VALENCES = ['virtue', 'vice']

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=10,
                    help='total substitutions per foundation/valence, seed words included')
parser.add_argument('--seed', type=int, default=20260813)
args = parser.parse_args()

df = pd.read_csv(TYPED_DICT)
# MFD 2.0's own spelling artifacts. They stay in the dictionary — as candidate
# *responses* they're legitimate, and the correctly-spelled variant of each is a
# separate entry anyway — but a misspelling in the prompt text would be an
# unintended confound, so they're barred from the substitution side.
MISSPELLINGS = {'abstinance', 'protecter', 'protecters', 'superviser', 'supervisers'}

pool = df[df['word_type'].isin(['noun', 'gerund']) & df['frame_fit']]
pool = pool[~pool['text'].isin(MISSPELLINGS)]
# Singular only. Plurals fill "I value ___ the most" perfectly well, but two of
# the templates continue "___ is the greatest thing ever", and "I think wounds
# is the greatest thing ever" doesn't parse. Rather than carry number agreement
# through the templates — which would vary the prompt wording between
# conditions, exactly what the matched baselines exist to avoid — every
# substitution is singular. There is comfortable supply: the tightest cell
# (loyalty.vice) has 9 singular candidates for the 6 slots.
pool = pool[pool['number'] == 'singular']
# `bm` is the only entry this drops: an abbreviation that doesn't read in the
# frame, unlike every other short entry in the dictionary.
pool = pool[pool['text'].str.len() > 2]

config = yaml.safe_load(SUBSTITUTIONS.read_text())['substitutions']
# Entries carrying a second tier are the seed words, which stay put and count
# toward each cell's N. Anything tagged only `sampled` is this script's own
# previous output, ignored so re-running regenerates the block from scratch.
seeds = [s for s in config if s['tiers'] != ['sampled']]
existing = {s['name'] for s in seeds}
seed_counts = collections.Counter(
    (s['foundation'], s['valence']) for s in seeds if 'sampled' in s['tiers'])

rng = random.Random(args.seed)
used = set(existing)
# Also block a chosen word's singular/plural twin, so a cell can't spend two of
# its ten slots on `overthrow` and `overthrows`.
used_lemmas = {wn.morphy(name, wn.NOUN) or name for name in existing}

print(f"  # --- Sampled from the dictionary — the non-seed members of the "
      f"`sampled` tier\n  # ({args.n} per foundation/valence including seeds, "
      f"random seed {args.seed}) ---")
for foundation in FOUNDATIONS:
    for valence in VALENCES:
        category = f"{foundation}.{valence}"
        n_needed = args.n - seed_counts[(foundation, valence)]
        candidates = sorted(pool[pool['categories'].apply(
            lambda c: category in c.split('|'))]['text'])
        candidates = [w for w in candidates if w not in used
                      and (wn.morphy(w, wn.NOUN) or w) not in used_lemmas]
        if len(candidates) < n_needed:
            raise ValueError(f"{category}: only {len(candidates)} candidates for n={n_needed}")

        chosen = []
        for word in rng.sample(candidates, len(candidates)):
            lemma = wn.morphy(word, wn.NOUN) or word
            if lemma in used_lemmas:
                continue  # twin drawn earlier in this same cell
            chosen.append(word)
            used.add(word)
            used_lemmas.add(lemma)
            if len(chosen) == n_needed:
                break
        if len(chosen) < n_needed:
            raise ValueError(f"{category}: only {len(chosen)} distinct-lemma candidates")
        chosen = sorted(chosen)

        print(f"\n  # {category} "
              f"(+ {seed_counts[(foundation, valence)]} seed words above)")
        for word in chosen:
            print(f"  - name: {word}")
            print(f"    foundation: {foundation}")
            print(f"    valence: {valence}")
            print(f"    tiers: [sampled]")
            print(f"    word_type: {pool.loc[pool['text'] == word, 'word_type'].iloc[0]}")
