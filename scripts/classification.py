from functools import lru_cache
from dotenv import load_dotenv
from pathlib import Path

import os
import pandas as pd
import pickle
import sys

sys.path.append(str(Path().absolute().parent))
from src.features.build_features import hashing_texts
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

load_dotenv()

CLASSES = os.getenv("CLASSES").split(",")
SUBSTITUTIONS = os.getenv("SUBSTITUTIONS").split(",")


@lru_cache
def load_models():
    path_models = Path().absolute().parent / "models"
    models = {}
    for class_ in CLASSES:
        models[class_] = pickle.load(
            open(
                path_models / f"{class_}_facts.pkl",
                "rb",
            )
        )
    return models


def main_classification_multiple_texts(
    texts: list, codigos: list, ids: list, output_path: str
):
    dic_models = load_models()
    results = []
    codigos_ = []
    ids_ = []
    for index, text in enumerate(texts):
        secoes = classifier_legal_sections_regex(text)
        if "fato" in secoes:
            # if "decisao" in secoes:
            results_dic = {k: 0 for k in dic_models.keys()}
            X = hashing_texts([secoes["fato"]])
            # X = hashing_texts([secoes["decisao"]])
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
                if (
                    class_ in row["codigos_movimentos_temas"].split(";")
                    and row[class_] > 0.5
                )
                or (
                    row[class_] <= 0.5
                    and class_ not in row["codigos_movimentos_temas"].split(";")
                )
                else 0
            )
        new_rows.append(dic_aux)
    df = pd.DataFrame(new_rows)
    df.to_csv(
        output_path + f"/validação_classes_teste.csv",
        # output_path + f"/validação_classes_teste_decisao.csv",
        index=False,
    )


def main_classification_multiple_texts_not_classified(
    texts: list, ids: list, output_path: str
):
    dic_models = load_models()
    results = []
    ids_ = []
    for index, text in enumerate(texts):
        secoes = classifier_legal_sections_regex(text)
        if "fato" in secoes:
            # if "decisao" in secoes:
            results_dic = {k: 0 for k in dic_models.keys()}
            X = hashing_texts([secoes["fato"]])
            # X = hashing_texts([secoes["decisao"]])
            for class_, model in dic_models.items():
                # results_dic[class_] = model.predict_proba(X)[0][1]
                results_dic[class_] = model.predict(X)[0]
            results.append(results_dic)
            ids_.append(str(ids[index]))
    df = pd.DataFrame(results)
    df["id_documento"] = ids_
    new_rows = []
    for row in df.to_dict("records"):
        dic_aux = row.copy()
        new_rows.append(dic_aux)
    df = pd.DataFrame(new_rows)
    df.to_csv(
        output_path + f"/classificação_decisões_cega.csv",
        index=False,
    )


if __name__ == "__main__":
    # VALIDAÇÃO DE MODELOS
    PATH_FILES_VALIDATE_MODELS = (
        Path().absolute().parent / "data/acordaos_principais_3k_3_F_limpo.csv"
    )
    OUTPUT_FILES_VALIDATE = str(Path().absolute().parent / "reports")
    print("Classificando e validando resultados")
    df_validation = pd.read_csv(
        PATH_FILES_VALIDATE_MODELS, encoding_errors="backslashreplace"
    )
    for s, c in zip(SUBSTITUTIONS, CLASSES):
        df_validation["codigos_movimentos_temas"] = df_validation[
            "codigos_movimentos_temas"
        ].str.replace(s, c, regex=False)
    main_classification_multiple_texts(
        df_validation["conteudo"].tolist(),
        df_validation["codigos_movimentos_temas"].tolist(),
        df_validation["numero_processo"].astype(str).tolist(),
        OUTPUT_FILES_VALIDATE,
    )

    # CLASSIFICAÇÃO DE TEXTOS NÃO CONHECIDOS
    PATH_FILES_CLASSIFY = Path().absolute().parent / "data/acordaos_principais_40k.csv"
    OUTPUT_FILES_CLASSIFY = str(Path().absolute().parent / "reports")
    print("Classificando processos desconhecidos")
    df = pd.read_csv(PATH_FILES_CLASSIFY)
    main_classification_multiple_texts_not_classified(
        df["conteudo"].tolist(),
        df["numero_processo"].astype(str).tolist(),
        OUTPUT_FILES_CLASSIFY,
    )
