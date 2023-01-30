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
    for class_ in ["85738", "85728", "85721", "85714", "85696", "85568"]:
        models[class_] = pickle.load(
            open(
                path_models
                / f"experiment_data_segmented_1_one_class_{class_}_facts_one_class.pkl",
                "rb",
            )
        )
    return models


def main_classification_multiple_texts(texts: list, codigos: list, ids: list):
    dic_models = load_models()
    results = []
    codigos_ = []
    ids_ = []
    for index, text in enumerate(texts):
        secoes = classifier_legal_sections_regex(text)
        if "fato" in secoes:
            results_dic = {k: 0 for k in dic_models.keys()}
            X = hashing_texts([secoes["fato"]], n_features=12000)
            for class_, model in dic_models.items():
                results_dic[class_] = model.predict_proba(X)[0][1]
            results.append(results_dic)
            codigos_.append(codigos[index])
            ids_.append(ids[index])
    df = pd.DataFrame(results)
    df["codigos_movimentos_temas"] = codigos_
    df["id_documento"] = ids_
    new_rows = []
    for row in df.to_dict("records"):
        dic_aux = row.copy()
        for class_ in dic_models:
            dic_aux[f"acertou_{class_}"] = (
                1
                if (class_ in row["codigos_movimentos_temas"] and row[class_] > 0.5)
                or (
                    row[class_] <= 0.5
                    and class_ not in row["codigos_movimentos_temas"]
                )
                else 0
            )
        new_rows.append(dic_aux)
    df = pd.DataFrame(new_rows)
    df.to_csv(f"D:\\TJSP_clustering_data\\acordaos_principais_3k_classes_results.csv")


if __name__ == "__main__":
    # dic_models = load_models()
    # example = "Julgo improcedente acordam em negar provimento"
    # secoes = classifier_legal_sections_regex(example)
    # if "fato" in secoes:
    #     X = hashing_texts([secoes["fato"]], n_features=12000)
    #     for class_, model in dic_models.items():
    #         print(f"Probabilidade de pertencimento à classe {class_}: {model.predict_proba(X)[0][1]}")

    df = pd.read_csv("D:\\TJSP_clustering_data\\acordaos_principais_3k.csv")
    main_classification_multiple_texts(
        df["conteudo"].tolist(),
        df["codigos_movimentos_temas"].tolist(),
        df["id_documento"].tolist(),
    )
