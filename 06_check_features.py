from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent

FILE = ROOT / "pose_features.csv"


print("="*70)
print("POSEFIT - CHECK FEATURES")
print("="*70)


df = pd.read_csv(FILE)


print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nCLASS DISTRIBUTION")
print(
    df["label"]
    .value_counts()
)


print("\nSPLIT DISTRIBUTION")

print(
    pd.crosstab(
        df["label"],
        df["split"]
    )
)


print("\nSOURCE DISTRIBUTION")

print(
    df["source"]
    .value_counts()
)


print("\nMissing values:")

print(
    df.isnull()
    .sum()
    .sum()
)


print("\nSample:")
print(
    df.head()
)


print("\nDONE")