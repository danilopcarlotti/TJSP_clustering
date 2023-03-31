# -*- coding: utf-8 -*-
"""
	
@author: Kelly
"""
"""
Exemplo:
    
clean_header_footer.py -s acordaos_processos.csv

"""
import os
import re
import pandas as pd
import argparse
import platform
from tqdm import tqdm

if platform.system() == "Windows":
    import winsound


def adjustFooterPattern(pattern):
    index = pattern.find("\n")
    if index != -1:
        return pattern[index:]

    return pattern


def detectHeader(s1, s2, minHeaderLen=20):
    i = 0
    n1 = len(s1)
    n2 = len(s2)
    j = minHeaderLen
    while j < n1:

        seq = s1[i:j]
        k = s2.find(seq)
        if k != -1:
            k += minHeaderLen
            while j < n1 and k < n2 and s1[j] == s2[k]:
                j += 1
                k += 1

            return s1[i:j]

        i += 1
        j += 1

    return ""


def detectFooter(s1, s2, minFooterLen=20):

    n1 = len(s1)
    # n2 = len(s2)
    i = n1 - minFooterLen
    j = n1
    while i >= 0:

        seq = s1[i:j]
        k = s2.rfind(seq)
        if k != -1:
            i -= 1
            k -= 1
            while i >= 0 and k >= 0 and s1[i] == s2[k]:
                i -= 1
                k -= 1

            return s1[(i + 1) : j]

        i -= 1
        j -= 1

    return ""


def removeHeaderFooter(s, pageSep="\x0c", maxHeaderLen=1000, maxFooterLen=1000):

    v = re.split(pageSep, s)

    pattern = ""

    numPages = len(v)

    for i in range(numPages):

        if i + 1 < numPages:
            page0 = "".join([c for c in v[i] if c != " " and c != "\n"])
            page1 = "".join([c for c in v[i + 1] if c != " " and c != "\n"])

            end0 = maxHeaderLen
            end1 = maxHeaderLen

            if end0 > len(page0):
                end0 = len(page0)

            if end1 > len(page1):
                end1 = len(page1)

            header = detectHeader(page0[:end0], page1[:end1])

            if header:
                pattern = r"\s*".join(list(header))

        if pattern:
            try:
                v[i] = re.sub(pattern, " ", v[i])
            except:
                pass

    pattern = ""

    for i in range(1, numPages):

        if i + 1 < numPages:
            page0 = "".join([c for c in v[i] if c != " "])
            page1 = "".join([c for c in v[i + 1] if c != " "])

            begin0 = len(page0) - maxFooterLen
            begin1 = len(page1) - maxFooterLen

            if begin0 < 0:
                begin0 = 0
            if begin1 < 0:
                begin1 = 0

            footer = detectFooter(page0[begin0:], page1[begin1:])

            if footer:
                footer = adjustFooterPattern(footer)
                footer = footer.strip()
                pattern = r"\s*".join(footer)

        if pattern:
            try:
                v[i] = re.sub(pattern, " ", v[i])
            except:
                pass

    return pageSep.join(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    requiredArgs = ap.add_argument_group("Argumento Obrigatório")
    requiredArgs.add_argument(
        "-s",
        "--source",
        help="especifica o arquivo a ser processado.",
        required=True,
        metavar="",
    )
    optionalArgs = ap.add_argument_group("Argumentos Opcionais")

    optionalArgs.add_argument(
        "-c",
        "--col",
        type=int,
        default=-1,
        help="especifica a coluna a ser processada. Se nenhuma "
        + " coluna for especificada, a coluna com os textos "
        + "será identificada de modo automático.",
        metavar="",
    )

    optionalArgs.add_argument(
        "-d",
        "--dest",
        default="",
        help="especifica o diretório para onde será salvo "
        + "o resultado do processamento. Se nenhum diretório "
        + "de destino for especificado, o resultado será salvo no "
        + "mesmo diretório do arquivo a ser processado.",
        metavar="",
    )
    optionalArgs.add_argument(
        "-i",
        type=int,
        default=0,
        help="especifica a partir de que linha do csv especificado "
        + "o processamento deve iniciar.",
        metavar="",
    )
    optionalArgs.add_argument(
        "-n",
        type=int,
        default=-1,
        help="especifica a quantidade de linhas a serem processadas. "
        + "Se nenhuma quantidade for especificada, todas as linhas "
        + "serão processadas, ignorando o argumento -i.",
        metavar="",
    )

    args = vars(ap.parse_args())

    df = pd.read_csv(args["source"])

    print(df)

    if args["col"] == -1:
        for i in range(len(df.columns)):
            if type(df.iloc[0, i]) is str and len(df.iloc[0, i]) > 500:
                args["col"] = i
                break

    firstRow = args["i"]

    if args["n"] != -1:
        lastRow = firstRow + args["n"]
    else:
        lastRow = firstRow + len(df.index)

    textCol = args["col"]

    n = lastRow - firstRow
    k = 1

    print(f"firstRow: {firstRow}, lastRow: {lastRow}, textCol: {df.columns[textCol]}")
    for i in tqdm(range(firstRow, lastRow)):
        try:

            tmp = removeHeaderFooter(df.iloc[i, textCol])

            if tmp != df.iloc[i, textCol]:
                print("diferente!")
            df.iloc[i, textCol] = tmp

            # print('\r progresso: ' + str(round(k*100/n,2)) + '%       ', end = "", flush = False)
            # k += 1
        except:
            pass

    if not args["dest"]:
        args["dest"] = os.path.dirname(args["source"])

    fileName = "header_and_footer_removed_" + os.path.basename(args["source"])

    df.to_csv(os.path.join(args["dest"], fileName), index=None)

    print(f"Output saved to: {os.path.join(args['dest'],fileName)}")
    if platform.system() == "Windows":
        winsound.Beep(500, 350)
    else:
        print("\a")
