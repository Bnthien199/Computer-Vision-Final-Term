"""Load validated bundles and save reproducible training/evaluation metadata."""
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from posefit_pipeline import LABELS, SPLITS


def load_bundle(path):
    path = Path(path)
    encoder = joblib.load(path / 'label_encoder.pkl')
    scaler = joblib.load(path / 'scaler.pkl')
    if set(encoder.classes_) != LABELS or scaler.n_features_in_ != 99:
        raise ValueError('Expected four PoseFit classes and 99 scaled xyz features')
    data = {}
    for split in SPLITS:
        x = np.load(path / f'X_{split}.npy', allow_pickle=False)
        y = np.load(path / f'y_{split}.npy', allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != 99 or not np.isfinite(x).all():
            raise ValueError(f'Invalid features: {split}')
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or set(y) != set(range(4)):
            raise ValueError(f'Invalid or missing classes: {split}')
        data[split] = (x, y)
    return data, encoder


def save_evaluation(y, pred, classes, output, split):
    labels = np.arange(len(classes))
    report = classification_report(y, pred, labels=labels, target_names=classes, zero_division=0, output_dict=True)
    report['accuracy'] = float(accuracy_score(y, pred))
    report['balanced_accuracy'] = float(balanced_accuracy_score(y, pred))
    report['confusion_matrix'] = confusion_matrix(y, pred, labels=labels).tolist()
    (output / f'{split}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{split}: accuracy={report["accuracy"]:.4f}, balanced={report["balanced_accuracy"]:.4f}, macro-F1={report["macro avg"]["f1-score"]:.4f}')


def save_run_info(output, data_dir, config):
    import importlib.metadata
    config['data_dir'] = str(Path(data_dir).resolve())
    config['versions'] = {name: importlib.metadata.version(name) for name in
                          ('numpy', 'scikit-learn', 'tensorflow', 'mediapipe')}
    (output / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    # A model's inference preprocessing travels with that model.
    for name in ('scaler.pkl', 'label_encoder.pkl'):
        shutil.copyfile(Path(data_dir) / name, output / name)
