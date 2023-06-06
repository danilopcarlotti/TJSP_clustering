from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

CLASSES = os.getenv("CLASSES").split(",")


def main_segmented(PATH_FILES):
    df = pd.read_csv(PATH_FILES)
    list_facts = df[df["type_section"] == "fato"].to_dict("records")[:]
    for c in df["class"].unique():
        c = str(c)
        if c not in CLASSES:
            continue
        print("Gerando dados da classe ", c)

        new_rows_facts = [
            {
                "text": row["text_section"],
                "id_text": row["id_text"],
                "class": 1 if str(int(row["class"])) == c else 0,
            }
            for row in list_facts
        ]

        df_all_facts = pd.DataFrame(new_rows_facts)
        df_all_facts = df_all_facts.sort_values(
            by="class", ignore_index=True, ascending=False
        )
        df_all_facts = df_all_facts.drop_duplicates(
            subset=["id_text"], keep="first", ignore_index=True
        )
        df_all_facts.drop(labels=["id_text"], inplace=True, axis=1)
        df_all_facts.reset_index(drop=True, inplace=True)
        df_all_facts.to_csv(PATH_FILES.replace(".csv", f"_{c}_facts.csv"), index=False)


if __name__ == "__main__":
    PATH_FILES = str(Path().absolute().parent / "data/base_principal.csv")
    main_segmented(PATH_FILES)
