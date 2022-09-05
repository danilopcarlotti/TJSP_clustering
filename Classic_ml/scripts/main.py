from pathlib import Path
import sys

from clustering_scoring import cluster_items

sys.path.append(str(Path().absolute().parent))

from src.models.train_model import cluster_documents, save_report_docx

if __name__ == "__main__":
    import pandas as pd

    # path = Path().absolute().parent / "data/texto.parquet"
    # path_classes = Path().absolute().parent / "data/classes.csv"

    # df_texts = pd.read_parquet(str(path))
    # df_classes = pd.read_csv(str(path_classes))

    # dict_data = {}
    # for row in df_texts.to_dict("records"):
    #     dict_data[row["idProcesso"]] = {"text":row["conteudoDocumento"].replace("\n", " "), "class":-1}
    # for row in df_classes.to_dict("records"):
    #     if row["idProcesso"] in dict_data:
    #         class_ = -1
    #         for c in df_classes.columns:
    #             if row[c] == 1:
    #                 class_ = c
    #                 break
    #         dict_data[row["idProcesso"]]["class"] = class_
    # new_rows = [v for v in dict_data.values()]
    # df_final = pd.DataFrame(new_rows)
    # df_final.to_csv(Path().absolute().parent / "data/experiment_data.csv")

    # PATH_FILE = sys.argv[1]
    # TYPE_FILE = sys.argv[2]
    # JUST_FINAL_DECISIONS = sys.argv[3]
    # OUTPUT_PATH = sys.argv[4]
    PATH_FILE = str(Path().absolute().parent / "data/experiment_data.csv")
    TYPE_FILE = "csv"
    JUST_FINAL_DECISIONS = 0
    OUTPUT_PATH = str(Path().absolute().parent / "reports/")

    args_dict = {
        "path_file": PATH_FILE,
        "type_file": TYPE_FILE,
        "just_final_decisions": JUST_FINAL_DECISIONS,
    }
    report = cluster_documents(args=args_dict)
    save_report_docx(report, output_path=OUTPUT_PATH)
