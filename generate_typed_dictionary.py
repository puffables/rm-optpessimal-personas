"""
Label every MFD 2.0 entry with its grammatical form and write
data/dictionaries/mfd2.0_typed.csv.

Why: the exhaustive search substitutes dictionary words into prompt frames
("I value ___ the most."), and only nominal entries survive that frame — the
dictionary mixes nouns, verbs, adjectives, adverbs and idioms freely
(`compassion`, `betray`, `loyal`, `lovingly`, `rock the boat` are all entries).
This produces the same `word_type` label that config/mfd_substitutions.yaml
carries, for the whole dictionary, so substitution sets can be sampled by form
and analysis can condition on it.

Two lexicons, because they fail in opposite places and neither is enough alone:

  BNC frequency list (data/corpora/1_1_all_alpha.txt, already in this repo for
  figures/generate_figure_3.R) gives a part of speech *with corpus frequencies*,
  which settles the genuinely ambiguous words — `care` is 396 noun vs 132 verb,
  `kill` is verb-dominant — but only covers the ~28% of MFD entries frequent
  enough to make the list.

  WordNet covers the rest, but only says what a word *can* be, not what it
  usually is. On its own it labels `kill` and `die` nouns (both have rare
  nominal senses), which is why BNC takes precedence where it has an opinion.
  The words BNC misses are mostly rare derivations (`nurturance`, `unkindness`)
  that are unambiguous anyway.

A context POS tagger was tried and rejected: these are bare dictionary entries
with no sentence around them, and NLTK's tagger defaults almost every isolated
word to NN — `betray`, `protect`, `loyal` and `unfair` all come back as nouns.

Labels, most-nominal-first (an entry gets the most nominal reading it supports,
since that's the reading the prompt frame forces):
    gerund     -ing form of a known verb (`suffering`, `cheating`)
    noun       noun-dominant in BNC, or noun-capable per WordNet
    verb       `betray`, `protect`, `obey`
    adjective  `loyal`, `sacred`, `unfair`
    adverb     `lovingly`, `safely`
Words in neither lexicon (rare inflections like `empathising`, agent nouns like
`nurturers`) fall through to suffix rules, so every entry gets a label.

Multiword entries are hand-labelled — no first-word or head-final rule survives
this set — into:
    gerund_phrase  (`being objective`, `stacking the deck`)
    verb_phrase    (`do unto others`, `rock the boat`, `go back on`)
    noun_phrase    (`team player`, `civil rights`, `golden rule`)
    other_phrase   idioms with no nominal head (`us against them`, `tit for tat`)

`frame_fit` marks the entries that can actually fill a bare "I value ___ the
most" slot: nominal, and either a gerund, an abstraction/mass noun, or a bare
plural, so singular countables (`hospital`, `child`) are excluded.

`number` records singular vs plural for nominal entries. Both fill the "I value
___" frame, but only singulars agree with the templates that continue "___ *is*
the greatest thing ever", so sample_mfd_substitutions.py draws from the
frame_fit singulars.

The CSV keeps the raw evidence (`bnc_pos`, `wordnet_pos`, `noun_supersense`)
next to the label so any individual call can be checked, and OVERRIDES holds the
handful the rules get wrong. Rerunning reproduces the file exactly.

Requires NLTK's WordNet corpus: python -c "import nltk; nltk.download('wordnet')"
"""
import collections
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

SCRIPT_ROOT = Path(__file__).parent
DICT_DIR = SCRIPT_ROOT / 'data' / 'dictionaries'
INPUT_PATH = DICT_DIR / 'mfd2.0.dic'
BNC_PATH = SCRIPT_ROOT / 'data' / 'corpora' / '1_1_all_alpha.txt'
OUTPUT_PATH = DICT_DIR / 'mfd2.0_typed.csv'

# The 110 multiword entries are labelled by hand rather than by rule. No
# first-word or head-final heuristic survives this set: `pay back` and `rock the
# boat` are verb phrases whose first word is noun-capable, while `team player`
# and `square deal` are noun phrases whose first word is verb-capable, and
# head-final fails on `take orders` and `pay back` in the other direction. The
# set is small, closed, and doesn't change unless the dictionary does.
PHRASE_LABELS = {
    # Noun phrases — the only multiword entries that can fill "I value ___".
    'civil rights': 'noun_phrase', 'civil right': 'noun_phrase',
    'golden rule': 'noun_phrase', 'square deal': 'noun_phrase',
    'square deals': 'noun_phrase', 'square dealer': 'noun_phrase',
    'square dealers': 'noun_phrase', 'due process': 'noun_phrase',
    'due processes': 'noun_phrase', 'due processing': 'noun_phrase',
    'level playing field': 'noun_phrase', 'level playing fields': 'noun_phrase',
    'economic disparity': 'noun_phrase', 'con artist': 'noun_phrase',
    'con artists': 'noun_phrase', 'false advertisement': 'noun_phrase',
    'false advertiser': 'noun_phrase', 'false advertisers': 'noun_phrase',
    'false impression': 'noun_phrase', 'false impressions': 'noun_phrase',
    'false witness': 'noun_phrase', 'double crosser': 'noun_phrase',
    'double crossers': 'noun_phrase', 'stacked deck': 'noun_phrase',
    'free rider': 'noun_phrase', 'free riders': 'noun_phrase',
    'team player': 'noun_phrase', 'union jack': 'noun_phrase',
    'old glory': 'noun_phrase', 'brothers in arms': 'noun_phrase',
    'divine right': 'noun_phrase', 'filial piety': 'noun_phrase',
    'social order': 'noun_phrase', 'pecking order': 'noun_phrase',
    'top gun': 'noun_phrase', 'top dog': 'noun_phrase',
    'prime minister': 'noun_phrase', 'prime ministers': 'noun_phrase',
    'corporate ladder': 'noun_phrase', 'corporate ladders': 'noun_phrase',
    'head honcho': 'noun_phrase', 'vice president': 'noun_phrase',
    'rabble rouser': 'noun_phrase', 'rabble rousers': 'noun_phrase',
    'trouble maker': 'noun_phrase', 'holy cross': 'noun_phrase',
    'holy crosses': 'noun_phrase', 'higher power': 'noun_phrase',

    # Verb phrases, including the inflected variants the dictionary lists
    # separately (`cheat on` / `cheats on` / `cheated on`).
    'do unto others': 'verb_phrase', 'pay back': 'verb_phrase',
    'will share': 'verb_phrase', 'will rob': 'verb_phrase',
    'did rob': 'verb_phrase', 'level the playing field': 'verb_phrase',
    'levels the playing field': 'verb_phrase', 'false advertise': 'verb_phrase',
    'false advertised': 'verb_phrase', 'false advertises': 'verb_phrase',
    'double cross': 'verb_phrase', 'double crosses': 'verb_phrase',
    'double crossed': 'verb_phrase', 'be partial': 'verb_phrase',
    'was partial': 'verb_phrase', 'am partial': 'verb_phrase',
    'been partial': 'verb_phrase', 'been objective': 'verb_phrase',
    'go back on': 'verb_phrase', 'rips off': 'verb_phrase',
    'ripped off': 'verb_phrase', 'stacks the deck': 'verb_phrase',
    'stacked the deck': 'verb_phrase', 'cheat on': 'verb_phrase',
    'cheats on': 'verb_phrase', 'cheated on': 'verb_phrase',
    'wear the crown': 'verb_phrase', 'do as one says': 'verb_phrase',
    'take orders': 'verb_phrase', 'take up arms': 'verb_phrase',
    'lead by example': 'verb_phrase', 'bow before': 'verb_phrase',
    'bow down': 'verb_phrase', 'preside over': 'verb_phrase',
    'lorded over': 'verb_phrase', 'toe the line': 'verb_phrase',
    'raise hell': 'verb_phrase', 'raises hell': 'verb_phrase',
    'rock the boat': 'verb_phrase',

    'being objective': 'gerund_phrase', 'being partial': 'gerund_phrase',
    'taking advantage': 'gerund_phrase', 'square dealing': 'gerund_phrase',
    'false advertising': 'gerund_phrase', 'double crossing': 'gerund_phrase',
    'stacking the deck': 'gerund_phrase', 'ripping off': 'gerund_phrase',
    'cheating on': 'gerund_phrase', 'rabble rousing': 'gerund_phrase',
    'raising hell': 'gerund_phrase',

    'prime ministerial': 'adjective_phrase',

    # Idioms and fragments with no nominal head.
    'tit for tat': 'other_phrase', 'eye for an eye': 'other_phrase',
    'behind their back': 'other_phrase', 'behind their backs': 'other_phrase',
    'death do us part': 'other_phrase', 'us against them': 'other_phrase',
    'all for one': 'other_phrase', 'one for all': 'other_phrase',
    'against us': 'other_phrase', 'by the book': 'other_phrase',
    'in charge': 'other_phrase',
}

# Single words the lexicons and suffix rules between them still get wrong.
OVERRIDES = {
    'fairminded': 'adjective',   # -ed suffix rule reads it as a past-tense verb
    'unbias': 'noun',            # in neither lexicon; MFD uses it nominally
    'proportional': 'adjective', # WordNet's lone rare noun sense outvotes the
                                 # everyday adjective under noun-first precedence
    'pities': 'verb',            # `pity` is noun-dominant, but the dictionary
                                 # lists the full verb paradigm (pities/pitied/
                                 # pitying) and this is its 3rd-person form
    'backstabs': 'verb',         # in neither lexicon; same verb paradigm
                                 # (backstab/backstabbed/backstabbing)
    'willing': 'adjective',      # -ing form of `will`, but MFD lists it in
                                 # authority.virtue beside `obedient` and
                                 # `compliant` — the adjective sense
}

# Suffix fallback for words in neither lexicon — mostly rare derivations
# (`nurturers`, `sluttiness`, `backstabbing`). Ordered: the first match wins, so
# more specific endings come first.
SUFFIX_RULES = [
    ('ing', 'gerund'),
    ('ly', 'adverb'),
    ('ier', 'adjective'),
    ('ness', 'noun'), ('ity', 'noun'), ('ment', 'noun'), ('tion', 'noun'),
    ('sion', 'noun'), ('ance', 'noun'), ('ence', 'noun'), ('ism', 'noun'),
    ('hood', 'noun'), ('ship', 'noun'),
    ('ers', 'noun'), ('er', 'noun'), ('ors', 'noun'), ('or', 'noun'),
    ('ists', 'noun'), ('ist', 'noun'),
    ('ed', 'verb'),
    ('al', 'adjective'), ('ic', 'adjective'), ('ous', 'adjective'),
    ('ful', 'adjective'), ('less', 'adjective'), ('y', 'adjective'),
]

# BNC tags: NoC = common noun, NoP = proper noun.
BNC_TO_LABEL = {'NoC': 'noun', 'NoP': 'noun', 'Verb': 'verb',
                'Adj': 'adjective', 'Adv': 'adverb'}


def load_bnc(path):
    """word -> {part of speech: frequency} from the BNC frequency list.

    Lemma rows carry the part of speech; the `@` rows that follow list that
    lemma's inflected forms, so inflections inherit their lemma's tag.
    """
    lexicon = collections.defaultdict(collections.Counter)
    current_pos = None
    for line in path.read_text(encoding='latin-1').splitlines():
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        word, tag, inflection, freq = (parts[1].strip(), parts[2].strip(),
                                       parts[3].strip(), parts[4].strip())
        if word == 'Word' or not freq.isdigit():
            continue
        if word == '@':
            if current_pos:
                lexicon[inflection][current_pos] += int(freq)
        else:
            current_pos = tag
            lexicon[word][tag] += int(freq)
    return lexicon


# Supersenses whose members are typically countable things rather than
# abstractions. A singular member of one of these can't fill a bare "I value ___
# the most" slot ("I value hospital the most"), though its plural can ("I value
# healers the most"), which is what frame_fit below encodes.
COUNTABLE_SUPERSENSES = {
    'noun.person', 'noun.artifact', 'noun.animal', 'noun.plant',
    'noun.location', 'noun.object', 'noun.body', 'noun.group', 'noun.food',
    'noun.Tops',
}

# Suffixes that mark a derived abstraction, for words WordNet doesn't have.
ABSTRACT_SUFFIXES = ('ness', 'ity', 'ment', 'tion', 'sion', 'ance', 'ence',
                     'ism', 'hood', 'ship', 'acy', 'ery')


def noun_supersense(word):
    """WordNet's semantic class for the word's most common noun sense.

    The first synset is WordNet's most frequent sense, which is the reading a
    bare word in a prompt frame will get.
    """
    synsets = wn.synsets(word.replace(' ', '_'), pos='n')
    return synsets[0].lexname() if synsets else ''


def is_plural(word, known_entries=frozenset()):
    """True if `word` is a plural noun.

    WordNet's morphy settles the words it knows. It fails two ways here: it
    returns nothing for words it doesn't know (`backstabbers`, `outgroups`), and
    it returns the word unchanged when the plural is *itself* a WordNet lemma
    (`prophets`, the biblical book). Either way it would pass the word off as
    singular, so both fall back to asking whether stripping the plural suffix
    leaves another entry of this same dictionary.
    """
    base = wn.morphy(word, wn.NOUN)
    if base and base != word:
        return True
    if word.endswith('ies') and word[:-3] + 'y' in known_entries:
        return True
    if word.endswith('es') and word[:-2] in known_entries:
        return True
    return word.endswith('s') and word[:-1] in known_entries


def grammatical_number(word, word_type, known_entries=frozenset()):
    """'singular', 'plural', or '' for non-nominal entries.

    Only nominals get a number. Gerunds and gerund phrases take singular
    agreement whatever their object ("being a team player *is* ..."), so they
    count as singular.
    """
    if word_type not in ('noun', 'noun_phrase', 'gerund', 'gerund_phrase'):
        return ''
    if word_type in ('gerund', 'gerund_phrase'):
        return 'singular'
    head = word.split()[-1]
    return 'plural' if is_plural(head, known_entries) else 'singular'


def fits_frame(word, word_type, known_entries=frozenset()):
    """Can this entry fill a bare nominal slot — "I value ___ the most"?

    True for abstractions and mass nouns (`compassion`, `filth`) and for bare
    plurals (`rights`, `healers`); False for singular countables (`hospital`,
    `child`) and for anything not nominal in the first place.

    Deliberately conservative: a few grammatical mass nouns that WordNet files
    under a countable supersense (`blood`, `flesh`) are excluded. That costs
    some candidates from a pool that has plenty, which is the right trade
    against sampling an entry that doesn't parse in the frame.
    """
    if word_type not in ('noun', 'gerund', 'noun_phrase', 'gerund_phrase'):
        return False
    # Gerunds are verbal nouns: always mass-like, always singular agreement
    # ("betraying is ..."), so they fit by construction. The supersense test
    # below would reject them, since WordNet files most -ing forms as events
    # or not at all.
    if word_type in ('gerund', 'gerund_phrase'):
        return True
    supersense = noun_supersense(word)
    if not supersense:
        return word.endswith(ABSTRACT_SUFFIXES) or is_plural(word, known_entries)
    return supersense not in COUNTABLE_SUPERSENSES or is_plural(word, known_entries)


def wordnet_pos(word):
    """Parts of speech WordNet knows for `word`, as a subset of {n, v, a, r}."""
    lookup = word.replace(' ', '_')
    return {pos for pos in ('n', 'v', 'a', 'r') if wn.synsets(lookup, pos=pos)}


def is_gerund(word):
    """True if `word` is the -ing form of a verb WordNet knows."""
    return word.endswith('ing') and bool(wn.morphy(word, wn.VERB))


def classify(entry, bnc):
    if entry in OVERRIDES:
        return OVERRIDES[entry]
    if ' ' in entry:
        return PHRASE_LABELS[entry]

    if is_gerund(entry):
        return 'gerund'

    # BNC first where it has an opinion: it knows which reading actually dominates.
    counts = bnc.get(entry)
    if counts:
        tag, _ = max(counts.items(), key=lambda kv: kv[1])
        if tag in BNC_TO_LABEL:
            return BNC_TO_LABEL[tag]

    pos = wordnet_pos(entry)
    # Noun-capable and verb-capable with no BNC frequencies to settle it: fall
    # back to how many senses WordNet records for each, on the lemma (so the
    # -s forms `segregates`, `tarnishes` are judged on `segregate`, `tarnish`).
    # Without this, capability alone calls every 3rd-person verb form a plural
    # noun. Ties go to the noun, matching the most-nominal-first precedence.
    if {'n', 'v'} <= pos:
        lemma_n = wn.morphy(entry, wn.NOUN) or entry
        lemma_v = wn.morphy(entry, wn.VERB) or entry
        noun_senses = len(wn.synsets(lemma_n, 'n'))
        verb_senses = len(wn.synsets(lemma_v, 'v'))
        # An -s form that is equally noun-ish and verb-ish is far more often the
        # 3rd-person verb than the plural in this dictionary (`disrespects`,
        # `pukes`), so ties go to the verb there and to the noun elsewhere.
        inflected = entry.endswith('s') and lemma_n != entry
        if verb_senses > noun_senses or (inflected and verb_senses >= noun_senses):
            return 'verb'

    # Verb-capable and adjective-capable with no noun reading: again decide by
    # sense count, so an archaic verb sense doesn't outrank the everyday
    # adjective (`compassionate` is 1 verb sense against 2 adjective senses).
    if {'v', 'a'} <= pos and 'n' not in pos:
        lemma_v = wn.morphy(entry, wn.VERB) or entry
        return ('verb' if len(wn.synsets(lemma_v, 'v')) > len(wn.synsets(entry, 'a'))
                else 'adjective')

    for key, label in (('n', 'noun'), ('v', 'verb'), ('a', 'adjective'), ('r', 'adverb')):
        if key in pos:
            return label

    for suffix, label in SUFFIX_RULES:
        if entry.endswith(suffix):
            return label
    # Nothing matched: a bare stem with no morphology (`fairplay`, `scuzz`),
    # which in this dictionary is always a noun.
    return 'noun'


def load_mfd(path):
    """Parse the LIWC-format .dic into [text, categories] (categories '|'-joined).

    The file is CR-terminated (classic Mac line endings); Python's universal
    newlines handles that. 63 words appear under two foundations, so entries are
    deduplicated by word with their categories merged.
    """
    lines = path.read_text().splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == '%']
    cat_names = dict(line.split('\t', 1) for line in lines[marks[0] + 1:marks[1]]
                     if line.strip())

    entries = {}
    for line in lines[marks[1] + 1:]:
        if not line.strip():
            continue
        word, *cat_ids = line.split('\t')
        entries.setdefault(word, []).extend(cat_names[c] for c in cat_ids)

    return pd.DataFrame({
        'text': list(entries),
        'categories': ['|'.join(dict.fromkeys(c)) for c in entries.values()],
    })


if __name__ == '__main__':
    bnc = load_bnc(BNC_PATH)
    df = load_mfd(INPUT_PATH)

    df['word_type'] = df['text'].map(lambda w: classify(w, bnc))
    df['bnc_pos'] = df['text'].map(
        lambda w: '|'.join(f"{tag}:{freq}" for tag, freq
                           in sorted(bnc.get(w, {}).items(), key=lambda kv: -kv[1])))
    df['wordnet_pos'] = df['text'].map(lambda w: '|'.join(sorted(wordnet_pos(w))))
    df['noun_supersense'] = df['text'].map(noun_supersense)
    entries = frozenset(df['text'])
    df['frame_fit'] = [fits_frame(t, wt, entries)
                       for t, wt in zip(df['text'], df['word_type'])]
    df['number'] = [grammatical_number(t, wt, entries)
                    for t, wt in zip(df['text'], df['word_type'])]
    df['n_words'] = df['text'].str.split().str.len()
    df = df[['text', 'categories', 'word_type', 'frame_fit', 'number', 'bnc_pos',
             'wordnet_pos', 'noun_supersense', 'n_words']]
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {OUTPUT_PATH} ({len(df)} entries)")
    print(df['word_type'].value_counts().to_string())
    pool = df[(df['word_type'] == 'noun') & df['frame_fit']]
    print(f"\n{len(pool)} nouns fit the prompt frame; per category "
          "(the pool substitutions are sampled from):")
    print(pool['categories'].str.split('|').explode().value_counts().to_string())
