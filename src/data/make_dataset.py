# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import sys

try:
    sys.path.append(str(Path().absolute().parent))
    from features.build_features import hashing_texts
except:
    sys.path.append(str(Path().absolute().parent.parent))
    from src.features.build_features import hashing_texts


def load_df(path_file: str, type_file: str):
    if type_file == "csv":
        df = pd.read_csv(path_file)
    elif type_file == "excel":
        df = pd.read_excel(path_file)
    elif type_file == "parquet":
        df = pd.read_parquet(path_file, engine="pyarrow")
    if "texto_publicacao" in df.columns:
        df.rename(columns={"texto_publicacao": "text"}, inplace=True)
    return df


def make_X_y(path_file: str, type_file: str):
    df = load_df(path_file, type_file)
    X = hashing_texts(df["text"])
    y = df["class"]
    return X, y
