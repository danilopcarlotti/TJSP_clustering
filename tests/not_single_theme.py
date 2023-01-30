from pathlib import Path
import os
import pandas as pd
import csv

import sys
sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")
from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

import assets

file_path='../data/base_acordaos2.parquet'
df = pd.read_parquet(file_path)
df.loc[:, 'number_of_themes'] = df[['S0929', 'S1015', 'S1033', 'S1039', 'S1046', 'S1101']].sum(axis='columns', numeric_only=True)

dg = df.loc[df['number_of_themes'] > 1]
dg["themes"] = dg.apply(lambda row: row[row == 1].index.tolist(), axis=1).values

print(dg)

with open(f"14k_fato_dual_theme.csv", "w") as fato:
    writer = csv.writer(fato)
    writer.writerow(["numero_processo", "numero_documento", "texto", "temas"])
    i=0
    for index, row in dg.iterrows():
        dic_data_text = classifier_legal_sections_regex(row["texto"])
        csv_row = [row['processo_id'], row['id'], dic_data_text["fato"] if "fato" in dic_data_text.keys() else "", ','.join(list(row['themes']))]
        writer.writerow(csv_row)
        print(i)
        i += 1


