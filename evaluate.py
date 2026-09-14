from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model import build_model


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "baseline_validation.txt"


def main():
    train = pd.read_csv(TRAIN_DATA_PATH, keep_default_na=False)

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        train["cleaned_review"],
        train["label_technical"],
        test_size=0.2,
        stratify=train["label_technical"],
        random_state=42,
    )

    model = build_model()
    model.fit(X_fit, y_fit)
    predictions = model.predict(X_validation)

    report = classification_report(
        y_validation,
        predictions,
        labels=[0, 1],
        target_names=["Other feedback", "Technical feedback"],
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    results = (
        "Baseline validation evaluation\n"
        "Source: data/processed/train.csv\n"
        "Split: 80/20, stratified, random_state=42\n"
        f"Training reviews: {len(X_fit)}\n"
        f"Validation reviews: {len(X_validation)}\n\n"
        f"{report}\n"
        "Confusion matrix: rows=actual, columns=predicted\n"
        "Class order: 0=Other feedback, 1=Technical feedback\n"
        f"{matrix}\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(results, encoding="utf-8")

    print(results)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()