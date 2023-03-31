from imblearn.over_sampling import SMOTE
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle
import re
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))

from src.data.make_dataset import make_X_y


def train_model(path_file: str, type_file: str, clf, output_path: str):
    X, y = make_X_y(path_file, type_file)
    calibrated_clf = CalibratedClassifierCV(clf, cv=10)
    calibrated_clf.fit(X, y)
    # clf.fit(X,y)
    pickle.dump(calibrated_clf, open(output_path, "wb"))

def plot_calibration(clf_list : list):
    fig = plt.figure(figsize=(11, 11))
    gs = GridSpec(6, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    colors = plt.cm.get_cmap("Dark2")
    names = []
    for i, (path, type_f, clf, _) in enumerate(clf_list):
        name = re.search(r"4k_(\d+)_fa", path).group(1)
        X, y = make_X_y(path, type_f)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # calibrated_clf = CalibratedClassifierCV(clf, cv=10)
        calibrated_clf = clf
        # sm = SMOTE(random_state=42)
        # X_train, y_train = sm.fit_resample(X_train, y_train)
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
    ax_calibration_curve.set_title("Calibration plots - NO SMOTE - no calibration")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
    for i, name in enumerate(names):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig("Calibration plots - NO SMOTE - no calibration.jpg")

if __name__ == "__main__":
    list_classifiers = [
        (
            "D:/TJSP_clustering_data/base3_1_14k_85568_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85568_facts.pkl",
        ),
        (
            "D:/TJSP_clustering_data/base3_1_14k_85696_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85696_facts.pkl",
        ),
        (
            "D:/TJSP_clustering_data/base3_1_14k_85714_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85714_facts.pkl",
        ),
        (
            "D:/TJSP_clustering_data/base3_1_14k_85721_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85721_facts.pkl",
        ),
        (
            "D:/TJSP_clustering_data/base3_1_14k_85728_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85728_facts.pkl",
        ),
        (
            "D:/TJSP_clustering_data/base3_1_14k_85738_facts.csv",
            "csv",
            LogisticRegression(max_iter=1000, random_state=42),
            "D:/TJSP_clustering_data/base3_1_14k_85738_facts.pkl",
        ),
    ]
    plot_calibration(list_classifiers)
    # for path, type_f, clf, output_path in list_classifiers:
    #     print("Saving model for dataset: ", path)
    #     train_model(path, type_f, clf, output_path)
