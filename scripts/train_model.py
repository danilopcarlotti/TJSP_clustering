from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os
import pickle
import re
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))

from src.data.make_dataset import make_X_y

load_dotenv()

possible_classes = os.getenv("CLASSES").split(",")


def train_model(
    path_file: str, type_file: str, clf, output_path: str, to_oversample: bool = False
):
    X, y = make_X_y(path_file, type_file)
    if to_oversample:
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
    calibrated_clf = CalibratedClassifierCV(clf, cv=2)
    calibrated_clf.fit(X, y)
    pickle.dump(calibrated_clf, open(output_path, "wb"))


def plot_calibration(clf_list: list, to_oversample: bool = False):
    fig = plt.figure(figsize=(11, 11))
    gs = GridSpec(2, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    colors = plt.cm.get_cmap("Dark2")
    for i, (path, type_f, clf, _) in enumerate(clf_list):
        name = re.search(r"_(\d+)_dec", path).group(1)
        X, y = make_X_y(path, type_f)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        calibrated_clf = CalibratedClassifierCV(clf, cv=2)
        if to_oversample:
            sm = SMOTE()
            X_train, y_train = sm.fit_resample(X_train, y_train)
        calibrated_clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            calibrated_clf,
            X_test,
            y_test,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots - KMEANS - calibration - weights")

    plt.tight_layout()
    plt.savefig(
        Path().absolute().parent
        / "reports/Calibration plots - KMEANS - calibration - weights.jpg"
    )


if __name__ == "__main__":

    list_classifiers = [
        (
            str(Path().absolute().parent / f"data/base_principal_{i}_decision.csv"),
            "csv",
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            Path().absolute().parent / f"models/{i}_decision.pkl",
        )
        for i in possible_classes
    ]
    plot_calibration(list_classifiers)

    for path, type_f, clf, output_path in list_classifiers:
        print("Saving model for dataset: ", path)
        train_model(path, type_f, clf, output_path)
