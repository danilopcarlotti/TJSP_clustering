from pathlib import Path
from scipy.spatial.distance import cosine as f_dist

import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path().absolute().parent))

from src.data.make_dataset import load_df
from src.features.build_features import hashing_texts


def statistics_arrays(path_file: str, type_file: str):
    df = load_df(path_file=path_file, type_file=type_file)
    df_decisions = df[df["class"] == "decisão"].copy(deep=True)
    df_petitions = df[df["class"] == "petição"].copy(deep=True)
    X = hashing_texts(df_decisions["text"].tolist())
    Y = hashing_texts(df_petitions["text"].tolist())
    rows = []
    for row_x in X:
        for row_y in Y:
            value = 1 - f_dist(row_x, row_y)
            rows.append(value)
    mean_sim = np.mean(rows)
    std_sim = np.std(rows)
    results = {
        "mean_sim_vectors": mean_sim,
        "lower_bound_sim": mean_sim - (2 * std_sim),
        "upper_bound_sim": mean_sim + (2 * std_sim),
    }
    return results


if __name__ == "__main__":
    PATH_FILE = sys.argv[1]
    TYPE_FILE = sys.argv[2]
    statistics_arrays(path_file=PATH_FILE, type_file=TYPE_FILE)
