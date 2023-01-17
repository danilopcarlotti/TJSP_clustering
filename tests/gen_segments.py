from pathlib import Path
import sys
sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")

print(sys.path)

from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex
import assets

import csv

data = assets.DataLoader(theme_id=sys.argv[1])

with open(f"14k_fato_{sys.argv[1]}.csv", "w") as fato:
    writer = csv.writer(fato)
    writer.writerow(["numero_processo", "numero_documento", "texto", "tema"])
    i=0
    for did, pid, theme, text in data:
        dic_data_text = classifier_legal_sections_regex(text)
        row = [pid, did, dic_data_text["fato"] if "fato" in dic_data_text.keys() else "", theme]
        writer.writerow(row)
        print(i)
        i += 1
        
        
