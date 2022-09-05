from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import HashingVectorizer

import nltk

from functools import lru_cache


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
