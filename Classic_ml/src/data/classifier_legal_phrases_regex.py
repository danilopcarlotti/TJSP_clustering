"""
PARSING AND SEGMENTATION OF SENTENCES IN SECTIONS
"""


from pathlib import Path

import os
import re
import spacy
import sys


sys.path.append(str(Path().absolute().parent.parent))
from src.data.regex_classifier_legal_phrases import words_interest


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