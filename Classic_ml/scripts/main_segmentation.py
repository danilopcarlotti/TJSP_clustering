from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))

from src.data.make_dataset import load_df
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

def segment_documents(args: dict):
    df = load_df(args["path_file"], args["type_file"])
    new_rows = []
    for row in df.to_dict("records"):
        dic_data_text = classifier_legal_sections_regex(row["text"])
        for k, v in dic_data_text.items():
            new_rows.append({
                "type_section":k,
                "text_section":v,
            })
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(Path().absolute().parent / "data/experiment_data_segmented.csv")

if __name__ == "__main__":
    import pandas as pd

    PATH_FILE = str(Path().absolute().parent / "data/experiment_data.csv")
    TYPE_FILE = "csv"
    JUST_FINAL_DECISIONS = 0
    OUTPUT_PATH = str(Path().absolute().parent / "reports/")

    args_dict = {
        "path_file": PATH_FILE,
        "type_file": TYPE_FILE,
        "just_final_decisions": JUST_FINAL_DECISIONS,
    }
    segment_documents(args=args_dict)
