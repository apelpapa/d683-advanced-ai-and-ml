from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = PROJECT_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_DIR / "models" / "tuned_model.joblib"
REPORT_PATH = PROJECT_DIR / "reports" / "final_test_evaluation.txt"
F1_TARGET = 0.75


def main():
    test = pd.read_csv(TEST_DATA_PATH, keep_default_na=False)

    X_test = test["cleaned_review"]
    y_test = test["label_technical"]

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Other feedback", "Technical feedback"],
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    technical_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        average="binary",
        zero_division=0,
    )

    target_status = "Met" if technical_f1 >= F1_TARGET else "Not met"

    results = (
        "Final held-out test evaluation\n"
        "Model: models/tuned_model.joblib\n"
        "Source: data/processed/test.csv\n"
        "The saved pipeline was used without refitting.\n"
        f"Test reviews: {len(test)}\n\n"
        f"{report}\n"
        "Confusion matrix: rows=actual, columns=predicted\n"
        "Class order: 0=Other feedback, 1=Technical feedback\n"
        f"{matrix}\n\n"
        f"Technical-feedback F1: {technical_f1:.4f}\n"
        f"F1 target: {F1_TARGET:.2f}\n"
        f"Target status: {target_status}\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(results, encoding="utf-8")

    print(results)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
