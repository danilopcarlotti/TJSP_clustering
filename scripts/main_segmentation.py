from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
import sys

load_dotenv()

CLASSES = os.getenv("CLASSES").split(",")
SUBSTITUTIONS = os.getenv("SUBSTITUTIONS").split(",")

sys.path.append(str(Path().absolute().parent))

from src.data.make_dataset import load_df
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex


def segment_documents(args: dict, multilabel: bool = False):
    df = load_df(args["path_file"], args["type_file"])
    df["codigos_movimentos_temas"] = df["codigos_movimentos_temas"].astype(str)
    for s, c in zip(SUBSTITUTIONS, CLASSES):
        df["codigos_movimentos_temas"] = df["codigos_movimentos_temas"].str.replace(
            s, c, regex=False
        )
    new_rows = []
    for index, row in enumerate(df.to_dict("records")):
        if not multilabel and ";" in row["codigos_movimentos_temas"]:
            continue
        dic_data_text = classifier_legal_sections_regex(row["conteudo"])
        for k, v in dic_data_text.items():
            if multilabel:
                for c in row["codigos_movimentos_temas"].split(";"):
                    new_rows.append(
                        {
                            "id_text": index,
                            "type_section": k,
                            "text_section": v,
                            "class": c,
                        }
                    )
            else:
                new_rows.append(
                    {
                        "id_text": index,
                        "type_section": k,
                        "text_section": v,
                        "class": row["codigos_movimentos_temas"],
                    }
                )
    if len(new_rows):
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(
            Path().absolute().parent / f"data/{args['dataset_name']}.csv", index=False
        )


if __name__ == "__main__":
    print("Segmentando base")
    OUTPUT_PATH = str(Path().absolute().parent / "reports/")
    TYPE_FILE = "csv"
    PATH_FILE = Path().absolute().parent / "data/base3_1_14k.csv"
    args_dict = {
        "path_file": PATH_FILE,
        "type_file": TYPE_FILE,
        "dataset_name": "base_principal_nova_seg",
    }
    segment_documents(args=args_dict, multilabel=True)
