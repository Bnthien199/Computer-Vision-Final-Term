"""Train the existing Random Forest baseline into a fresh run directory."""
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib
from posefit_pipeline import ROOT, new_output
from posefit_training import load_bundle, save_evaluation, save_run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'training_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/randomforest')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=1, help='Worker count; 1 also works in restricted Windows environments')
    args = parser.parse_args()
    data, encoder = load_bundle(args.data_dir)
    output = new_output(args.output, directory=True)
    model = RandomForestClassifier(n_estimators=300, max_depth=None,
                                   random_state=args.seed, class_weight='balanced', n_jobs=args.jobs)
    model.fit(*data['train'])
    joblib.dump(model, output / 'randomforest_posefit.pkl')
    for split in ('val', 'test'):
        x, y = data[split]
        save_evaluation(y, model.predict(x), encoder.classes_, output, split)
    save_run_info(output, args.data_dir, dict(model='RandomForest', seed=args.seed,
                                            n_estimators=300, class_weight='balanced', jobs=args.jobs))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
