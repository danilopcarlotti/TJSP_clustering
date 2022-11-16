from pathlib import Path
from functools import lru_cache

import pickle
import sys

sys.path.append(str(Path().absolute().parent))
from src.features.build_features import hashing_texts

@lru_cache
def load_models():
    path_models = Path().absolute().parent / "models"
    models = {}
    for class_ in ["0929", "1015", "1033", "1039", "1046", "1101"]:
        models[class_] = pickle.load(open(path_models / f"experiment_data_segmented_1_S{class_}_facts.pkl", "rb"))
    return models

if __name__ == "__main__":
    dic_models = load_models()
    example = "Julgo improcedente acordam em negar provimento"
    X = hashing_texts([example], n_features=12000)
    for class_, model in dic_models.items():
        print(f"Probabilidade de pertencimento à classe {class_}: {model.predict_proba(X)[0][1]}")
