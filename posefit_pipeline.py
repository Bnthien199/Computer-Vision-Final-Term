"""Schema validation and non-overwriting output paths for PoseFit."""
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [f'f_{i}' for i in range(99)]
LABELS = {'plank', 'pushup', 'squat', 'standing'}
SPLITS = ('train', 'val', 'test')


def new_output(path, directory=False):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'runs'):
        raise ValueError('Output must be inside the project runs/ directory')
    if path.exists():
        raise FileExistsError(f'Output already exists; choose a new run: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        path.mkdir()
    return path


def validate_metadata(df):
    required = {'filepath', 'label', 'source', 'subject', 'split'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f'Metadata must be nonempty and contain {sorted(required)}')
    if df[['filepath', 'label', 'source', 'split']].isna().any().any():
        raise ValueError('Missing required metadata values')
    if not set(df.label).issubset(LABELS) or not set(df.split).issubset(SPLITS):
        raise ValueError('Unsupported labels or splits')
    if not set(df.source).issubset({'gym', 'standing'}):
        raise ValueError('Unsupported source')
    if ((df.source == 'standing') != (df.label == 'standing')).any():
        raise ValueError('Standing source and label must agree')
    validate_subject_split(df)
    if not df.filepath.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Source paths must be nonempty strings')
    paths = [(ROOT / value).resolve() for value in df.filepath]
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated source path (possibly across splits)')
    for path in paths:
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing or out-of-project source: {path}')


def validate_features(df, require_all_splits=True):
    columns = [c for c in df if c.startswith('f_')]
    if set(columns) != set(FEATURE_COLUMNS) or df.empty:
        raise ValueError('Expected nonempty features with exactly f_0..f_98')
    if not {'label', 'split'}.issubset(df.columns) or df[['label', 'split']].isna().any().any():
        raise ValueError('Missing label/split')
    if not set(df.label).issubset(LABELS) or not set(df.split).issubset(SPLITS):
        raise ValueError('Unsupported feature labels or splits')
    if require_all_splits and (set(df.label) != LABELS or set(df.split) != set(SPLITS)):
        raise ValueError('Expected all four labels and train/val/test splits')
    values = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('NaN/Inf in landmark coordinates')
    for split in SPLITS if require_all_splits else ():
        if set(df.loc[df.split.eq(split), 'label']) != LABELS:
            raise ValueError(f'Every split needs all classes: {split}')
    if 'filepath' in df:
        if not df.filepath.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError('Missing feature source paths')
        paths = df.filepath.map(lambda value: os.path.normcase(str((ROOT / value).resolve())))
        if (df.groupby(paths).split.nunique() > 1).any():
            raise ValueError('Missing provenance or source paths shared across splits')
        if (df.groupby(paths).label.nunique() > 1).any():
            raise ValueError('Conflicting action labels for the same source')
        if 'frame' in df:
            frames = df.frame.to_numpy(dtype=float)
            if not np.isfinite(frames).all() or np.any(frames < 0) or np.any(frames != np.floor(frames)):
                raise ValueError('Frame indices must be finite nonnegative integers')
            if df.assign(_source=paths).duplicated(['_source', 'frame']).any():
                raise ValueError('Repeated frame from the same source')
    validate_subject_split(df)
    return values


def validate_subject_split(df):
    # Only standing has subject annotations. Do not invent identities for gym.
    if 'subject' in df:
        standing = df.loc[df.label.eq('standing')]
        if standing.subject.isna().any():
            raise ValueError('Standing samples need subject IDs')
        subjects = standing.subject.map(str)
        if (standing.groupby(subjects).split.nunique() > 1).any():
            raise ValueError('Standing subject shared across splits')
