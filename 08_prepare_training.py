import argparse
from posefit_pipeline import new_output, validate_metadata, validate_features, FEATURE_COLUMNS
from pathlib import Path
from posefit_console import configure_console
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# =================================================
# PATH
# =================================================

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "pose_features.csv"

OUTPUT = ROOT / "runs/prepared"

# =================================================
# LOAD DATA
# =================================================

def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/prepared")
    args = parser.parse_args()
    print("="*70)
    print("POSEFIT - PREPARE TRAINING DATA")
    print("="*70)

    df = pd.read_csv(args.input)
    validate_features(df)

    print("\nDataset:")
    print(df.shape)

    # =================================================
    # FEATURES
    # =================================================

    feature_cols = FEATURE_COLUMNS

    X = df[feature_cols].values

    y = df["label"].values

    print("\nFeature:")
    print(X.shape)

    # =================================================
    # ENCODE LABEL
    # =================================================

    encoder = LabelEncoder()

    encoder.fit(df.loc[df["split"].eq("train"), "label"])
    y_encoded = encoder.transform(y)

    print("\nClasses:")

    for i, c in enumerate(
        encoder.classes_
    ):
        print(
            i,
            "->",
            c
        )

    # =================================================
    # SPLIT
    # =================================================

    train_idx = (
        df["split"]
        ==
        "train"
    )

    val_idx = (
        df["split"]
        ==
        "val"
    )

    test_idx = (
        df["split"]
        ==
        "test"
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]
    y_test = y_encoded[test_idx]

    # =================================================
    # NORMALIZATION
    # =================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    # =================================================
    # SAVE
    # =================================================

    OUTPUT = new_output(args.output, directory=True)

    np.save(
        OUTPUT / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT / "X_val.npy",
        X_val
    )

    np.save(
        OUTPUT / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT / "y_val.npy",
        y_val
    )

    np.save(
        OUTPUT / "y_test.npy",
        y_test
    )

    joblib.dump(
        encoder,
        OUTPUT / "label_encoder.pkl"
    )

    joblib.dump(
        scaler,
        OUTPUT / "scaler.pkl"
    )

    print("\nSaved training data")

    print("\nTRAIN:")
    print(X_train.shape)

    print("VAL:")
    print(X_val.shape)

    print("TEST:")
    print(X_test.shape)

    print("\nDONE")

    for split in ("train", "val", "test"):
        df.loc[df["split"].eq(split), [c for c in df if c not in feature_cols]].to_csv(
            OUTPUT / f"samples_{split}.csv", index=False)
    if "filepath" not in df:
        print("WARNING: legacy CSV lacks source filepath; clip leakage cannot be audited.")

if __name__ == "__main__":
    main()
