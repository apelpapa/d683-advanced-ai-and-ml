import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "tuned_model.joblib"


def predict_reviews(input_path, output_path):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The tuned model is missing. Run tune.py first."
        )

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose a new filename."
        )

    data = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=False,
        encoding="utf-8-sig",
    )

    if "cleaned_review" not in data.columns:
        raise ValueError(
            "The input CSV must contain a 'cleaned_review' column."
        )

    if data.empty:
        raise ValueError("The input CSV contains no reviews.")

    reviews = (
        data["cleaned_review"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if reviews.eq("").any():
        raise ValueError(
            "The input CSV contains blank reviews in 'cleaned_review'."
        )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(reviews)

    results = pd.DataFrame(
        {
            "cleaned_review": data["cleaned_review"],
            "predicted_label": predictions,
        }
    )

    results["predicted_category"] = results["predicted_label"].map(
        {
            0: "Other feedback",
            1: "Technical feedback",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        mode="x",
    )

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Classify game reviews as technical or other feedback."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV containing a cleaned_review column.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for a new predictions CSV.",
    )

    args = parser.parse_args()

    try:
        review_count = predict_reviews(args.input, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))

    print(f"Reviews classified: {review_count}")
    print(f"Predictions saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()