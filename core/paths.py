"""
Repo-wide paths and data-file helpers.

Scripts in experiment folders bootstrap the import of this module with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N = depth below repo root
    from core.paths import ROOT, CONFIG_DIR, DATA_DIR

Notebooks find the repo root by walking up from the working directory until
they reach a folder containing core/paths.py (see find_repo_root()).
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'


def find_repo_root(start=None):
    """Walk up from `start` (default: cwd) to the folder containing core/paths.py."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'core' / 'paths.py').is_file():
            return candidate
    raise FileNotFoundError(f"No repo root (core/paths.py) found above {start}")


def load_config(name):
    """Load config/<name> (YAML)."""
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def active_reward_models():
    """Reward models to (re)score: entries in config/reward_models.yaml without `active: false`.

    Analyses of already-scored data (the paper figures/tables) use every entry
    via load_config('reward_models.yaml') instead.
    """
    return [m for m in load_config('reward_models.yaml') if m.get('active', True)]


def safe_model_name(model_name):
    """'org/model' -> 'org--model', the filename stem used for every per-model CSV."""
    return model_name.replace('/', '--')


def scores_csv(directory, model_name):
    """Path of a per-model score/logit CSV, accepting either a plain or gzipped copy.

    The large persona CSVs are tracked in git only as <model>.csv.gz; the
    generators write plain <model>.csv checkpoints while running. When both
    exist, the more recently modified one wins, so a fresh `git pull` of the
    .gz isn't shadowed by an older local .csv. Returns the plain .csv path
    (which may not exist) when neither is present, so callers can write to it.
    """
    plain = Path(directory) / f"{safe_model_name(model_name)}.csv"
    gz = plain.with_name(plain.name + '.gz')
    if plain.exists() and gz.exists():
        return plain if plain.stat().st_mtime >= gz.stat().st_mtime else gz
    if gz.exists():
        return gz
    return plain
