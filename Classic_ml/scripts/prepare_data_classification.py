import pandas as pd


def main_segmented(PATH_FILES):
    df = pd.read_csv(PATH_FILES)
    list_facts = df[df["type_section"] == "fato"].to_dict("records")[:]
    for c in df["class"].unique():
        print("Gerando dados da classe ", c)

        new_rows_facts = [
            {"text": row["text_section"], "class": 1 if row["class"] == c else 0}
            for row in list_facts
        ]

        df_all_facts = pd.DataFrame(new_rows_facts)
        df_all_facts.to_csv(
            PATH_FILES.replace(".csv", f"_{c}_facts.csv"), index=False
        )


def main_not_segmented():
    PATH_FILES = "D:\\TJSP_clustering_data\\experiment_data_one_class.csv"
    df = pd.read_csv(PATH_FILES)
    list_values = df.to_dict("records")[:]
    for c in df["class"].unique():
        print("Gerando dados da classe ", c)
        new_rows_all = [
            {"text": row["text"], "class": 1 if row["class"] == c else 0}
            for row in list_values
        ]

        df_all_class = pd.DataFrame(new_rows_all)
        df_all_class.to_csv(PATH_FILES.replace(".csv", f"_{c}.csv"), index=False)


if __name__ == "__main__":
    PATH_FILES = "D:/TJSP_clustering_data/base3_1_14k.csv"
    main_segmented(PATH_FILES)

    PATH_FILES = "D:/TJSP_clustering_data/base3_1_3k.csv"
    main_segmented(PATH_FILES)
