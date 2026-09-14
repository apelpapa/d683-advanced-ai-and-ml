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

    return cleaned.reset_index(drop=True)

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
train = pd.read_csv(RAW_DATA_DIR / "train.csv", keep_default_na=False)
test = pd.read_csv(RAW_DATA_DIR / "test.csv", keep_default_na=False)

train = clean_dataset(train)
test = clean_dataset(test)

for name, dataSet in [("Training", train), ("Testing", test)]:
    print(f"\n{name} dataset")
    print(f"Rows: {len(dataSet)}")
    print(f"Columns: {dataSet.columns.tolist()}")
    print("Technical label counts:")
    print(dataSet["label_technical"].value_counts().sort_index())
