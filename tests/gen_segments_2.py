import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")

from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex
from tqdm import tqdm
import csv

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

file_path = sys.argv[1]
tqdm.pandas()

df = pd.read_parquet(file_path)

tmp = (
    df["conteudo"]
    .parallel_apply(classifier_legal_sections_regex)
    .parallel_apply(pd.Series)
)

df = df.join(tmp)

df.to_parquet(f"{sys.argv[1][:-12]}_with_segments.parquet.gzip", compression="gzip")
