import pandas as pd

PATH_FILES = "D:/TJSP_clustering_data/experiment_data_segmented_1_one_class.csv"

df = pd.read_csv(PATH_FILES)
list_values = df.to_dict("records")[:]
list_facts = df[df["type_section"] == "fato"].to_dict("records")[:]
list_law = df[df["type_section"] == "lei"].to_dict("records")[:]
for c in df["class"].unique():
    print("Gerando dados da classe ", c)
    new_rows_all = [
        {"text": row["text_section"], "class": 1 if row["class"] == c else 0}
        for row in list_values
    ]
    new_rows_facts = [
        {"text": row["text_section"], "class": 1 if row["class"] == c else 0}
        for row in list_facts
    ]
    new_rows_law = [
        {"text": row["text_section"], "class": 1 if row["class"] == c else 0}
        for row in list_law
    ]

    df_all_class = pd.DataFrame(new_rows_all)
    df_all_class.to_csv(
        PATH_FILES.replace(".csv", f"_{c}_all_one_class.csv"), index=False
    )

    df_all_facts = pd.DataFrame(new_rows_facts)
    df_all_facts.to_csv(
        PATH_FILES.replace(".csv", f"_{c}_facts_one_class.csv"), index=False
    )

    df_all_law = pd.DataFrame(new_rows_law)
    df_all_law.to_csv(
        PATH_FILES.replace(".csv", f"_{c}_law_one_class.csv"), index=False
    )
