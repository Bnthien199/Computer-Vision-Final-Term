import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent

df = pd.read_csv(
    ROOT / "pose_features.csv"
)


feature_cols = [
    c for c in df.columns
    if c.startswith("f_")
]


print("Landmark missing:")
print(
    df[feature_cols]
    .isnull()
    .sum()
    .sum()
)


print("\nMetadata missing:")
print(
    df[
        [
            "label",
            "source",
            "split",
            "subject",
            "frame"
        ]
    ]
    .isnull()
    .sum()
)