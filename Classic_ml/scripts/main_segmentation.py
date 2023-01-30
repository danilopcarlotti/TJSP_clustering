from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))

from src.data.make_dataset import load_df
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex


def segment_documents(args: dict):
    df = load_df(args["path_file"], args["type_file"])
    # df = df.sample(n=5000)
    new_rows = []
    counter = 1
    for index, row in enumerate(df.to_dict("records")):
        if ";" in row["codigos_movimentos_temas"]:
            continue
        dic_data_text = classifier_legal_sections_regex(row["conteudo"])
        for k, v in dic_data_text.items():
            new_rows.append(
                {
                    "id_text": index,
                    "type_section": k,
                    "text_section": v,
                    # "class":row["codigos_movimentos_temas"].split(";")[0],
                    "class": row["codigos_movimentos_temas"],
                }
            )
            if len(new_rows) > 500000:
                new_df = pd.DataFrame(new_rows)
                new_df.to_csv(
                    f"D:\\TJSP_clustering_data\\no_classes_experiment_data_segmented_{counter}_one_class.csv"
                )
                new_rows = []
                counter += 1
    if len(new_rows):
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(
            f"D:\\TJSP_clustering_data\\experiment_data_segmented_{counter}_one_class.csv"
        )


if __name__ == "__main__":
    import pandas as pd

    PATH_FILE = "D:\\TJSP_clustering_data\\acordaos_principais_14k.csv"
    TYPE_FILE = "csv"
    JUST_FINAL_DECISIONS = 0
    OUTPUT_PATH = str(Path().absolute().parent / "reports/")

    args_dict = {
        "path_file": PATH_FILE,
        "type_file": TYPE_FILE,
        "just_final_decisions": JUST_FINAL_DECISIONS,
    }
    segment_documents(args=args_dict)
