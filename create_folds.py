from ast import arg
import os
import argparse
import pandas as pd
from sklearn import model_selection

from constants import DATA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-folds", type=int, help="Number of folds", default=5)
    args = parser.parse_args()

    # Load dataset
    data_path = os.path.join(DATA_PATH, "train.csv")
    df = pd.read_csv(data_path)

    # Shuffle
    df["kfold"] = -1
    df = df.sample(frac=1.0).reset_index(drop=True)

    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=args.num_folds)

    # Create folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X=df, y=y)):
        pass
