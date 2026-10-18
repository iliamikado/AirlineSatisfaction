from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_baseline.cbm"

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    data = pd.read_csv(train_path)
    print(f"Loaded {len(data):,} training rows and {data.shape[1]} columns.")

    # `id` identifies a row, but does not describe a passenger.
    X = data.drop(columns=[TARGET, ID_COLUMN])
    y = data[TARGET].astype(int)

    # CatBoost natively handles categorical columns, but they must not be NaN.
    categorical_columns = X.select_dtypes(include="object").columns.tolist()
    X[categorical_columns] = X[categorical_columns].fillna("Missing")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=1_000,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
    )

    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_probabilities)
    print(f"\nValidation ROC-AUC: {auc:.6f}")
    print(f"Best iteration: {model.get_best_iteration()}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
