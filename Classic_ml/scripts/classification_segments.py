from functools import lru_cache
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
import pandas as pd
import pickle
import re
import spacy


@lru_cache
def stopwords_pt():
    return set(nltk.corpus.stopwords.words("portuguese"))


@lru_cache
def remove_accents(text):
    accents = {
        "Á": "A",
        "Ã": "A",
        "À": "A",
        "á": "a",
        "ã": "a",
        "à": "a",
        "É": "E",
        "é": "e",
        "Ê": "E",
        "ê": "e",
        "Í": "I",
        "í": "i",
        "Ó": "O",
        "ó": "o",
        "Õ": "O",
        "õ": "o",
        "Ô": "O",
        "ô": "o",
        "Ú": "U",
        "ú": "u",
        ";": "",
        ",": "",
        "/": "",
        "\\": "",
        "{": "",
        "}": "",
        "(": "",
        ")": "",
        "-": "",
        "_": "",
        "Ç": "C",
        "ç": "c",
    }
    text = str(text)
    for k, v in accents.items():
        text = text.replace(k, v)
    return text


def normalize_texts(texts, to_stem=False):
    if to_stem:
        stemmer = nltk.stem.RSLPStemmer()
    normal_texts = []
    tk = RegexpTokenizer(r"\w+")
    stopwords = stopwords_pt()
    for t in texts:
        raw_text = remove_accents(t.lower())
        tokens = tk.tokenize(raw_text)
        processed_text = ""
        for tkn in tokens:
            if tkn.isalpha() and tkn not in stopwords and len(tkn) > 3:
                if to_stem:
                    tkn = stemmer.stem(tkn)
                processed_text += tkn + " "
        normal_texts.append(processed_text[:-1])
    return normal_texts


def hashing_texts(texts, n_features=15000, ngram_range=(1, 1)):
    corpus = normalize_texts(texts)
    return (
        HashingVectorizer(n_features=n_features, ngram_range=ngram_range)
        .fit_transform(corpus)
        .toarray()
    )


words_interest = {
    "[\s,]lei[\s,]": "lei",
    "s[úu]mula": "lei",
    "artigo": "lei",
    "c[óo]digo": "lei",
    "aduz": "fato",
    "afirma": "fato",
    "argumenta": "fato",
    "assever": "fato",
    "constata-se": "fato",
    "deduz": "fato",
    "esclarece": "fato",
    "laudo": "fato",
    "no caso": "fato",
    "per[íi]cia": "fato",
    "prova": "fato",
    "sustenta": "fato",
    "trata\-se": "fato",
    "absolv": "decisao",
    "acolhimento": "decisao",
    "acordam": "decisao",
    "arquive": "decisao",
    "celeb.{1,10}acordo": "decisao",
    "conced": "decisao",
    "conden": "decisao",
    "dar.{,10}provimento": "decisao",
    "determino": "decisao",
    "expeça": "decisao",
    "extingo": "decisao",
    "proced.ncia": "decisao",
    "procedente": "decisao",
    "provido": "decisao",
    "provimento": "decisao",
    "homolog.{1,10}acordo": "decisao",
    "[\s,]pede[\s,]": "pedido",
    "[\s,]pedido[\s,]": "pedido",
    "pleitea": "pedido",
    "pretende": "pedido",
    "seja condenado": "pedido",
    "seja deferido": "pedido",
    "[\s,]solicita[\s,]": "pedido",
}


def break_sentences(text, nlp):
    "BREAKS TEXT IN SENTENCES"
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"art\.", "art ", text)
    text = re.sub(r"fls?\.", "fls ", text)
    text = re.sub(r"inc\.", "inc ", text)
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def dictionary_phrases_classes():
    "REGEX FOR CLASSIFYING PHRASES WITH EACH SECTION"
    dic_tipos_frases = {}
    for frase, tipo in words_interest.items():
        if tipo not in dic_tipos_frases:
            dic_tipos_frases[tipo] = []
        dic_tipos_frases[tipo].append(r"{}".format(frase))
    return dic_tipos_frases


def classifier_legal_sections_regex(text):
    "CLASSIFIER OF SECTIONS"
    nlp = spacy.load("pt_core_news_sm")
    dic_tipos_frases = dictionary_phrases_classes()
    secoes = {}
    for tipo, conj_exp in dic_tipos_frases.items():
        for frase in break_sentences(text, nlp):
            for exp in conj_exp:
                if re.search(exp, frase, flags=re.I | re.S):
                    if tipo not in secoes:
                        secoes[tipo] = ""
                    secoes[tipo] += " " + frase
                    break
    return secoes


@lru_cache
def load_models():
    path_models = Path().absolute().parent / "models"
    models = {}
    for class_ in ["85738", "85728", "85721", "85714", "85696", "85568"]:
        models[class_] = pickle.load(
            open(
                path_models
                / f"experiment_data_segmented_1_one_class_{class_}_facts_one_class.pkl",
                "rb",
            )
        )
    return models


def load_models_no_segments():
    path_models = Path().absolute().parent / "models"
    models = {}
    for class_ in ["85738", "85728", "85721", "85714", "85696", "85568"]:
        models[class_] = pickle.load(
            open(
                path_models / f"experiment_data_one_class_{class_}.pkl",
                "rb",
            )
        )
    return models


def main_classification_multiple_texts_not_classified(df: pd.DataFrame):
    texts = df["conteudo"].tolist()
    ids = df["numero_processo"].tolist()
    dic_models = load_models()
    results = []
    ids_ = []
    conteudos = []
    for index, conteudo in enumerate(texts):
        secoes = classifier_legal_sections_regex(conteudo)
        if "fato" in secoes:
            results_dic = {k: 0 for k in dic_models.keys()}
            X = hashing_texts([secoes["fato"]], n_features=12000)
            for class_, model in dic_models.items():
                results_dic[class_] = model.predict_proba(X)[0][1]
            results.append(results_dic)
            ids_.append(str(ids[index]))
            conteudos.append(conteudo)
    df = pd.DataFrame(results)
    df["numero_processo"] = ids_
    df["conteudo"] = conteudos
    new_rows = []
    for row in df.to_dict("records"):
        dic_aux = row.copy()
        new_rows.append(dic_aux)
    df = pd.DataFrame(new_rows)
    return df


if __name__ == "__main__":
    print("Classificando acórdãos da fila")
    df = pd.read_csv("D:\\TJSP_clustering_data\\acordaos_principais_fila.csv")
    df_classification = main_classification_multiple_texts_not_classified(df)
    df_classification.to_csv("D:\\TJSP_clustering_data\\acordaos_principais_fila_results.csv", index=False)

    print("Classificando acórdãos dos 40k")
    df = pd.read_csv("D:\\TJSP_clustering_data\\acordaos_principais_40k.csv")
    df_classification = main_classification_multiple_texts_not_classified(df)
    df_classification.to_csv("D:\\TJSP_clustering_data\\acordaos_principais_fila_results.csv", index=False)
