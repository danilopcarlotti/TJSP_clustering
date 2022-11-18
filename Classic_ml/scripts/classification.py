from pathlib import Path
from functools import lru_cache

import pandas as pd
import pickle
import sys

sys.path.append(str(Path().absolute().parent))
from src.features.build_features import hashing_texts
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

@lru_cache
def load_models():
    path_models = Path().absolute().parent / "models"
    models = {}
    for class_ in ["0929", "1015", "1033", "1039", "1046", "1101"]:
        models[class_] = pickle.load(open(path_models / f"experiment_data_segmented_1_S{class_}_facts.pkl", "rb"))
    return models


def main_classification_multiple_texts(texts : list):
    dic_models = load_models()
    results = []
    for text in texts:
        secoes = classifier_legal_sections_regex(text)
        if "fato" in secoes:
            results_dic = {k:0 for k in dic_models.keys()}
            X = hashing_texts([secoes["fato"]], n_features=12000)
            for class_, model in dic_models.items():
                results_dic[class_] = model.predict_proba(X)[0][1]
            results.append(results_dic)
    df = pd.DataFrame(results)
    df.to_csv(f"D:\\TJSP_clustering_data\\no_classes_results.csv")
    print(df.describe())

if __name__ == "__main__":
    # dic_models = load_models()
    # example = "Julgo improcedente acordam em negar provimento"
    # secoes = classifier_legal_sections_regex(example)
    # if "fato" in secoes:
    #     X = hashing_texts([secoes["fato"]], n_features=12000)
    #     for class_, model in dic_models.items():
    #         print(f"Probabilidade de pertencimento à classe {class_}: {model.predict_proba(X)[0][1]}")

    df = pd.read_csv("D:\\TJSP_clustering_data\\acordaos_sem_tema.csv")
    main_classification_multiple_texts(df["conteudo"].tolist())
