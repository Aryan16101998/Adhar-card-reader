# THIS IS THE FILE WHICH READS ALL text FROM CSV FILE 
import csv
import sys 
path = './csv_file/'+sys.argv[1]
with open(path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)