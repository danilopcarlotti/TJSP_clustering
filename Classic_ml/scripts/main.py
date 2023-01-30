from pathlib import Path
import re
import sys

from clustering_scoring import cluster_items

sys.path.append(str(Path().absolute().parent))

from src.models.train_model import cluster_documents, save_report_docx

if __name__ == "__main__":
    # import pandas as pd

    # path = "D:\\TJSP_clustering_data\\base_acordaos2.parquet"
    # df_texts = pd.read_parquet(str(path))
    # new_rows = []
    # for row in df_texts.to_dict("records"):
    #     for c in ['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101']:
    #         if row[c] == 1:
    #             new_rows.append({"text":re.sub(r"\s+", " ", row["texto"]), "class":c})
    # df_final = pd.DataFrame(new_rows)
    # df_final.to_csv("D:\\TJSP_clustering_data\\experiment_data.csv", index=False)

    # # PATH_FILE = sys.argv[1]
    # # TYPE_FILE = sys.argv[2]
    # # JUST_FINAL_DECISIONS = sys.argv[3]
    # # OUTPUT_PATH = sys.argv[4]

    PATH_FILE = str(Path().absolute().parent / "data/acordaos_principais_14k.csv")
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
