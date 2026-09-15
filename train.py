from pathlib import Path

import joblib
import pandas as pd

from model import build_model

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_DIR / "models" / "baseline_model.joblib"


def main():
    train = pd.read_csv(TRAIN_DATA_PATH, keep_default_na=False)

    X_train = train["cleaned_review"]
    y_train = train["label_technical"]

    model = build_model()
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    vocabulary_size = len(model.named_steps["tfidf"].vocabulary_)

    print("Training complete.")
    print(f"Training reviews: {len(X_train)}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Learned classes: {model.classes_.tolist()}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
