from pathlib import Path

import pandas as pd


def clean_dataset(dataset):
    required_columns = ["cleaned_review", "label_technical"]

    missing_columns = set(required_columns) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    cleaned = dataset[required_columns].copy()

    cleaned["cleaned_review"] = (
        cleaned["cleaned_review"]
        .astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned["label_technical"] = pd.to_numeric(
        cleaned["label_technical"], errors="coerce"
    )

    valid_text = cleaned["cleaned_review"].ne("")
    valid_label = cleaned["label_technical"].isin([0, 1])

    cleaned = cleaned.loc[valid_text & valid_label].copy()
    cleaned["label_technical"] = cleaned["label_technical"].astype(int)

    review_keys = cleaned["cleaned_review"].str.casefold()

    conflicting_labels = cleaned.groupby(review_keys)["label_technical"].nunique() > 1
    if conflicting_labels.any():
        raise ValueError("Duplicate reviews have conflicting technical labels.")

    duplicate_rows = review_keys.duplicated(keep="first")
    print(f"Within-dataset duplicates removed: {duplicate_rows.sum()}")

    cleaned = cleaned.loc[~duplicate_rows].copy()

    return cleaned.reset_index(drop=True)


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
train = pd.read_csv(RAW_DATA_DIR / "train.csv", keep_default_na=False)
test = pd.read_csv(RAW_DATA_DIR / "test.csv", keep_default_na=False)

train = clean_dataset(train)
test = clean_dataset(test)

train_keys = train["cleaned_review"].str.casefold()
test_keys = test["cleaned_review"].str.casefold()

overlapping_reviews = train_keys.isin(test_keys)
print(f"Training reviews overlapping with test: {overlapping_reviews.sum()}")

train = train.loc[~overlapping_reviews].copy().reset_index(drop=True)

for name, dataSet in [("Training", train), ("Testing", test)]:
    print(f"\n{name} dataset")
    print(f"Rows: {len(dataSet)}")
    print(f"Columns: {dataSet.columns.tolist()}")
    print("Technical label counts:")
    print(dataSet["label_technical"].value_counts().sort_index())

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

train.to_csv(
    PROCESSED_DATA_DIR / "train.csv",
    index=False,
    encoding="utf-8",
)

test.to_csv(
    PROCESSED_DATA_DIR / "test.csv",
    index=False,
    encoding="utf-8",
)

print(f"\nPreprocessed datasets saved to: {PROCESSED_DATA_DIR}")
