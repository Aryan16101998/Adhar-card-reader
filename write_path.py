############################ Writing path in csv file #########################

import argparse as ap
import csv 
import sys




fr = ['./csv_file/'+'addhar_input.csv', './csv_file/'+'video_inout.csv']
# try:
print(sys.argv)
# fileopen = open('./csv_file/test.csv', 'w')
if sys.argv[1] == '1':
    file_path = fr[0]
    fileopen = open(file_path, 'w')
    print(file_path, '== write == ', sys.argv[2])
    writer = csv.writer(fileopen)
    writer.writerow(['aadhar_front_image','aadhar_back_image'])
    writer.writerow([sys.argv[2], sys.argv[3]])
    fileopen.close()

if sys.argv[1] == '2':
    file_path = fr[1]
    fileopen = open(file_path, 'w')
    print(file_path, '== write == ', sys.argv[2], end='')
    writer = csv.writer(fileopen)
    writer.writerow([sys.argv[2]])