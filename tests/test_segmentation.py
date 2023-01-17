from pathlib import Path
import sys
sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")

print(sys.path)

from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex
import assets

data = assets.DataLoader(theme_id="S0929")

for _, (did, pid, theme, text) in zip(range(3), data):
    dic_data_text = classifier_legal_sections_regex(text)
    
    print(f"\033[41mid:\033[0m {did}\n\033[41mprocesso_id:\033[0m {pid}\n\033[41mtheme:\033[0m {theme}\n")
    
    for k, v in dic_data_text.items():
        print(f"\033[41m{k}:\033[0m \"{v}\" \n")
    break
