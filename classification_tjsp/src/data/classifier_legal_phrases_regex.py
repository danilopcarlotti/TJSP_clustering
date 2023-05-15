"""
PARSING AND SEGMENTATION OF SENTENCES IN SECTIONS
"""


from pathlib import Path

import re
import spacy
import sys


sys.path.append(str(Path().absolute().parent.parent))
from src.data.regex_classifier_legal_phrases import words_interest, words_connections


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
        phrases = []
        for index, frase in enumerate(break_sentences(text, nlp)):
            found = False
            for exp in conj_exp:
                if re.search(exp, frase, flags=re.I | re.S):
                    if tipo not in secoes:
                        secoes[tipo] = ""
                    secoes[tipo] += " " + frase
                    phrases.append(1)
                    found = True
                    break
            if not found:
                phrases.append(0)
                found_connection = False
                for exp_c in words_connections:
                    if re.search(exp_c, frase[:20], flags=re.I | re.S):
                        found_connection = True
                if index > 0 and phrases[index - 1] == 1 and found_connection:
                    secoes[tipo] += " " + frase
    return secoes
