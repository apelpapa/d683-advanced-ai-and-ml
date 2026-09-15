import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from model import build_model


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models" / "tuned_model.joblib"
N_FOLDS = 5


def main():
    train = pd.read_csv(TRAIN_DATA_PATH, keep_default_na=False)
    baseline_scores = pd.read_csv(REPORTS_DIR / "baseline_cv_folds.csv")

    X = train["cleaned_review"]
    y = train["label_technical"]

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42,
    )

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
        "classifier__class_weight": [None, "balanced"],
    }

    scorer = make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0,
    )

    search = GridSearchCV(
        estimator=build_model(),
        param_grid=parameter_grid,
        scoring=scorer,
        cv=cv,
        refit=True,
        n_jobs=4,
        verbose=2,
        error_score="raise",
    )

    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("rank_test_score")

    baseline_f1 = float(baseline_scores["f1"].mean())
    best_f1 = float(search.best_score_)
    best_std = float(search.cv_results_["std_test_score"][search.best_index_])

    summary = {
        "selection_metric": "technical_feedback_f1",
        "training_reviews": len(train),
        "folds": N_FOLDS,
        "shuffle": True,
        "random_state": 42,
        "candidate_count": len(results),
        "parameter_grid": parameter_grid,
        "baseline_cv_f1": baseline_f1,
        "best_cv_f1": best_f1,
        "best_cv_f1_std": best_std,
        "cv_f1_improvement": best_f1 - baseline_f1,
        "best_parameters": search.best_params_,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORTS_DIR / "tuning_results.csv",
        index=False,
        encoding="utf-8",
    )

    summary_text = json.dumps(summary, indent=2)
    summary_path = REPORTS_DIR / "tuning_summary.json"
    summary_path.write_text(summary_text, encoding="utf-8")

    joblib.dump(search.best_estimator_, MODEL_PATH, compress=3)

    print("\nTuning complete.")
    print(summary_text)
    print(f"\nTuned model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
