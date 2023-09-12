import csv
import random
import sys

def shuffle_tsv(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        rows = list(reader)
    
    header, data = rows[0], rows[1:]
    random.shuffle(data)

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    tsv_filename = sys.argv[1]
    shuffle_tsv(tsv_filename)
