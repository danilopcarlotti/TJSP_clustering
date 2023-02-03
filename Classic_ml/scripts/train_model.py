from pathlib import Path
from sklearn.linear_model import LogisticRegression

import pickle
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))

from src.data.make_dataset import make_X_y


def train_model(path_file: str, type_file: str, clf, output_path: str):
    X, y = make_X_y(path_file, type_file)
    clf.fit(X, y)
    pickle.dump(clf, open(output_path, "wb"))


if __name__ == "__main__":
    for path, type_f, clf, output_path in [
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85568.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85568.pkl",
        ),
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85696.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85696.pkl",
        ),
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85714.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85714.pkl",
        ),
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85721.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85721.pkl",
        ),
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85728.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85728.pkl",
        ),
        (
            "D:/TJSP_clustering_data/experiment_data_one_class_85738.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/experiment_data_one_class_85738.pkl",
        ),
    ]:
        print("Saving model for dataset: ", path)
        train_model(path, type_f, clf, output_path)
