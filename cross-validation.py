from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from model import build_model

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
N_FOLDS = 5


def main():
    train = pd.read_csv(TRAIN_DATA_PATH, keep_default_na=False)

    X = train["cleaned_review"]
    y = train["label_technical"]

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    scores = cross_validate(
        build_model(),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    fold_scores = pd.DataFrame(
        {
            "fold": range(1, N_FOLDS + 1),
            "accuracy": scores["test_accuracy"],
            "precision": scores["test_precision"],
            "recall": scores["test_recall"],
            "f1": scores["test_f1"],
        }
    )

    metric_columns = ["accuracy", "precision", "recall", "f1"]
    summary = pd.DataFrame(
        {
            "mean": fold_scores[metric_columns].mean(),
            "std": fold_scores[metric_columns].std(ddof=0),
        }
    )

    report = (
        "Baseline stratified cross-validation\n"
        "Source: data/processed/train.csv\n"
        f"Reviews: {len(train)}\n"
        f"Folds: {N_FOLDS}, shuffle=True, random_state=42\n"
        "Precision, recall, and F1 measure technical feedback (class 1).\n\n"
        f"{fold_scores.round(4).to_string(index=False)}\n\n"
        "Mean and standard deviation across folds (ddof=0):\n"
        f"{summary.round(4).to_string()}\n"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_scores.to_csv(
        REPORTS_DIR / "baseline_cv_folds.csv",
        index=False,
        encoding="utf-8",
    )

    report_path = REPORTS_DIR / "baseline_cv_summary.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
