from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_baseline.cbm"
SUBMISSION_PATH = PROJECT_DIR / "outputs" / "submission.csv"

ID_COLUMN = "id"


def main() -> None:
    test_path = DATA_DIR / "test.csv"
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun `python train.py` first."
        )
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test = pd.read_csv(test_path)
    ids = test[ID_COLUMN]
    X_test = test.drop(columns=[ID_COLUMN])
    categorical_columns = X_test.select_dtypes(include="object").columns.tolist()
    X_test[categorical_columns] = X_test[categorical_columns].fillna("Missing")

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({ID_COLUMN: ids, "satisfaction": probabilities})
    SUBMISSION_PATH.parent.mkdir(exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission with {len(submission):,} rows saved to: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
