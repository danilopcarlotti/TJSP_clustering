from pathlib import Path

import sys
import pandas as pd

sys.path.append(str(Path().absolute().parent))

from src.data.make_dataset import load_df


def extract_documents(args: dict):
    df = load_df(args["path_file"], args["type_file"])
    new_rows = []
    for row in df.to_dict("records"):
        if ";" in row["codigos_movimentos_temas"]:
            continue
        new_rows.append(
            {
                "text":row["conteudo"],
                "class": row["codigos_movimentos_temas"],
            }
        )
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(f"D:\\TJSP_clustering_data\\experiment_data_one_class.csv")

if __name__ == "__main__":
    

    PATH_FILE = "D:\\TJSP_clustering_data\\acordaos_principais_14k.csv"
    TYPE_FILE = "csv"
    JUST_FINAL_DECISIONS = 0
    OUTPUT_PATH = str(Path().absolute().parent / "reports/")

    args_dict = {
        "path_file": PATH_FILE,
        "type_file": TYPE_FILE,
        "just_final_decisions": JUST_FINAL_DECISIONS,
    }
    extract_documents(args=args_dict)
