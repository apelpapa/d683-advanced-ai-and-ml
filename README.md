## D683 – ADVANCED AI AND ML

Welcome to Advanced AI and ML!

## Requirements

### Software

The application was developed on Windows 11 Pro (64-bit) using Python 3.14.7 with the following packages:

|     Package     |   Version    |
|:---------------:|:------------:|
|   cloudpickle   |    3.1.2     |
|     joblib      |    1.6.0     |
|    narwhals     |    2.26.0    |
|      numpy      |    2.5.3     |
|     pandas      |    3.0.5     |
|  pandas-stubs   | 3.0.5.260914 |
| python-dateutil | 2.9.0.post0  |
|  scikit-learn   |    1.9.1     |
|      scipy      |    1.18.1    |
|       six       |    1.17.0    |
|  threadpoolctl  |    3.6.0     |
|     tzdata      |    2026.4    |

This list records the tested Python environment, including supporting dependencies and development tools.

A Python virtual environment is recommended to isolate these packages from other projects.

### Hardware

The application uses CPU processing and does not require a GPU.

It was developed and tested on a computer with:

- Intel Core i9-12900KS Processor
- 128 GB RAM

These are the tested hardware specifications. Minimum CPU and RAM requirements have not been benchmarked.

The application requires write access to the project directory to save processed data, trained models, and reports.

### Data

Preprocessing requires the supplied CSV files:

- 'data/raw/train.csv'
- 'data/raw/test.csv'

Both files must contain these columns:

- 'cleaned_review': review text
- 'label_technical': 1 for technical feedback or 0 for other feedback

The preprocessing script generates the files used for training, validation, and tuning under 'data/processed'.

## Installation and Execution

### Set up the environment

Install Python 3.14.7 (64-bit)

Open PowerShell in the project's root folder and run the following commands in this folder.

Verify the Python Version:

```
python --version
```

Create a Virtual Environment:

```
python -m venv .venv
```

Install the required packages:

```
$projectPackages = @(
    "cloudpickle==3.1.2",
    "joblib==1.6.0",
    "narwhals==2.26.0",
    "numpy==2.5.3",
    "pandas==3.0.5",
    "pandas-stubs==3.0.5.260914",
    "python-dateutil==2.9.0.post0",
    "scikit-learn==1.9.1",
    "scipy==1.18.1",
    "six==1.17.0",
    "threadpoolctl==3.6.0",
    "tzdata==2026.4"
)

.\.venv\Scripts\python.exe -m pip install $projectPackages
```

### Prepare the Data

Ensure the supplied training and testing CSVs are located in `data/raw`, then run:

```
.\.venv\Scripts\python.exe preprocess.py
```

This creates `data/processed/train.csv` and `data/processed/test.csv`.

### Train the baseline model

```
.\.venv\Scripts\python.exe train.py
```

This trains the baseline pipeline and saves it to `models/baseline_model.joblib`.

### Evaluate the baseline on a validation split

```
.\.venv\Scripts\python.exe evaluate.py
```

Accuracy, precision, recall, F1, and the confusion matrix are saved to `reports/baseline_validation.txt`.

### Run cross-validation

```
.\.venv\Scripts\python.exe cross-validation.py
```

This saves individual fold scores to `reports/baseline_cv_folds.csv` and their summary to
`reports/baseline_cv_summary.txt`.

### Tune the model

Run cross-validation before tuning because the tuning script reads its saved baseline scores.

```
.\.venv\Scripts\python.exe tune.py
```

This compares 32 configurations using five-fold stratified cross-validation and selects the highest mean
technical-feedback F1 score.

### Output

The script saves:

- `models/tuned_model.joblib`: the fitted selected pipeline
- `reports/tuning_results.csv`: scores for every configuration
- `reports/tuning_summary.json`: selected settings and comparison with the baseline

### Evaluate the final model

After tuning, run:

```
.\.venv\Scripts\python.exe evaluate_final.py
```

This loads the saved tuned pipeline and evaluates it on the held-out test reviews without retraining.

Accuracy, precision, recall, F1, the confusion matrix, and the target status are saved to
`reports/final_test_evaluation.txt`.

The final model achieved an accuracy of 0.8350 and a technical-feedback F1 score of 0.6972. The F1 target of 0.75 was
not met.

### Classify new reviews

After installing the dependencies, use the saved tuned model
to classify new reviews. Training is only necessary if the
saved model needs to be rebuilt.

The input must be a UTF-8 CSV containing a `cleaned_review`
column and at least one review. Every review must contain
nonblank text. Existing classification labels are not required.
Reviews containing commas or line breaks must be enclosed
in double quotes.

To classify the included example reviews, run this command
from the project root:

```powershell
.\.venv\Scripts\python.exe predict.py --input data/new_reviews.csv --output reports/predictions.csv
```

Replace the input and output paths to use other files.

The output CSV contains:

- `cleaned_review`: the original input text
- `predicted_label`: 1 for technical feedback or 0 for other feedback
- `predicted_category`: Technical feedback or Other feedback

The application reports errors for missing files, a missing
review column, blank reviews, or an existing output file.
Choose a new output filename for each run.

To display command-line help:

```powershell
.\.venv\Scripts\python.exe predict.py --help
```

## Data Attribution

This project uses Version 1 of the Steam Review Aspect Dataset by Sandy Khosasi (2024), available on GitHub and
described in the original article. The dataset is licensed under CC BY 4.0.
The processed data retains the cleaned review text and technical-feedback label, normalizes whitespace, and removes one
training review duplicated in the test set.
- Dataset: [GitHub](https://github.com/ilos-vigil/steam-review-aspect-dataset)
- Original article: [Steam Review Aspect Dataset](https://srec.ai/blog/steam-review-aspect-dataset)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)