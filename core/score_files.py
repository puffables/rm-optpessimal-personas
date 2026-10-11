"""
Reading and writing the per-model score/logit CSVs that the generator scripts
checkpoint to (paper/generate_*.py, persona/generate_*.py).

Every such CSV has the columns token_id, token_name, token_decoded, followed by
one column per prompt (or template__persona combination).
"""
import gzip
import shutil
from pathlib import Path

import pandas as pd

TOKEN_COLUMNS = ['token_id', 'token_name', 'token_decoded']


def vocab_frame(tokenizer):
    """token_id / token_name / token_decoded for the tokenizer's full vocabulary, by id."""
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids],
                                           skip_special_tokens=False)
    return pd.DataFrame({
        'token_id': token_ids,
        'token_name': token_names,
        'token_decoded': token_decoded,
    })


def existing_columns(path):
    """Score columns already present in a checkpoint CSV (empty set if none yet)."""
    path = Path(path)
    if not path.exists():
        return set()
    return set(pd.read_csv(path, nrows=0).columns) - set(TOKEN_COLUMNS)


def load_checkpoint(path, vocab_df):
    """Start a run from vocab_df, carrying over any already-scored columns from `path`.

    Old scores are joined on token_id rather than assigned by row position, so a
    checkpoint written with a different tokenizer revision (e.g. one that has
    since added a [PAD] token) can't silently shift scores onto the wrong tokens.
    The token text columns always come from the current tokenizer, which also
    repairs files written by older versions of these scripts with
    escapechar='\\' (which doubled every backslash in token_decoded on read).
    """
    path = Path(path)
    if not path.exists():
        return vocab_df.copy()
    old = pd.read_csv(path)
    score_cols = [c for c in old.columns if c not in TOKEN_COLUMNS]
    df = vocab_df.merge(old[['token_id'] + score_cols], on='token_id', how='left')
    missing = len(set(old['token_id']) - set(vocab_df['token_id']))
    if missing:
        print(f"  Warning: {missing} token_ids in {path.name} are not in the current vocabulary "
              f"and were dropped")
    return df


def save_checkpoint(df, path):
    """Write a checkpoint as plain CSV (default quoting; no escapechar)."""
    df.to_csv(path, index=False)


def gzip_copy(path):
    """Write <path>.gz next to a finished CSV: the form the large files are tracked in."""
    path = Path(path)
    gz_path = path.with_name(path.name + '.gz')
    with open(path, 'rb') as src, gzip.GzipFile(gz_path, 'wb', compresslevel=9, mtime=0) as dst:
        shutil.copyfileobj(src, dst)
    return gz_path
