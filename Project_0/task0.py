import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "task0_sl19d1"
    TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
    TEST_PATH = os.path.join(DATA_PATH, "test.csv")
    SUBMISSION_PATH = "submission.csv"

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    x_cols = [c for c in test.columns if c.startswith("x")]

    train_mean = train[x_cols].mean(axis=1)
    mae = (train["y"] - train_mean).abs().mean()

    print(f"Train check: mean absolute error vs rule y=mean(x): {mae:.6f}")

    RMSE = mean_squared_error(train["y"], train_mean)**0.5
    print("Train RMSE:", RMSE)

    y_pred = test[x_cols].mean(axis=1)


    submission = pd.DataFrame({"Id": test["Id"], "y": y_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Wrote {SUBMISSION_PATH} with {len(submission)} rows.")
    print(submission.head())