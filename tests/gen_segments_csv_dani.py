from pathlib import Path
import sys, os
from multiprocessing import Process, Queue
from queue import Empty
sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent) + "/Classic_ml/")

print(sys.path)

from src.data.classifier_legal_phrases_regex import classifier_legal_sections_regex

import csv
csv.field_size_limit(10000000)

NUM_PROCESSES = 8

def seg(qi, qr):
    while True:
        try:
            row = qi.get(timeout=5)
        except Empty:
            print(f"Processo {os. getpid()} terminou!")
            return    
        
        dic_data_text = classifier_legal_sections_regex(row[2])
        row[2] = dic_data_text["fato"] if "fato" in dic_data_text.keys() else ""
        print(row[0])
        qr.put(row)

def load(qi, reader):
    for row in reader:
        qi.put(row)
    print("Leitura da entrada finalizada!")
        
with open(sys.argv[1], 'r') as csv_dani, open("../data/tmp.csv", "w") as fato:
    reader = csv.reader(csv_dani)
    writer = csv.writer(fato)
    header = next(reader, None)
    header[2] = "fato"
    writer.writerow(header)
    
    qi = Queue(maxsize=128)
    qr = Queue(maxsize=128)
    
    p_list = []
    p = Process(target=load, args=(qi, reader))
    p_list.append(p)
    p.start()
    
    for i in range(NUM_PROCESSES):
        p = Process(target=seg, args=(qi, qr))
        p_list.append(p)
        p.start()
    
    while True:
        try:
            row = qr.get(timeout=5)
        except Empty:
            print("Done !")
            break 
        writer.writerow(row)
    
    for p in p_list:
        p.join()
        
