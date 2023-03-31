import csv
import sys

csv.field_size_limit(100000000)

with open(f"{sys.argv[1]}", "r") as fato:
    reader = csv.reader(fato)
    l = []
    for row in reader:
        print(row[0])
        l.append(row[0])

    print(len(l))
