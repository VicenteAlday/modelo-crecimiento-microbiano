import random

with open(r'C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets\dataset.csv', 'r') as r, open('shuffleddataset.csv', 'w') as w:
    data = r.readlines()
    header, rows = data[0], data[1:]
    random.shuffle(rows)
    rows = '\n'.join([row.strip() for row in rows])
    w.write(header + rows)