import csv

csvfile = open('output.csv')
reader = csv.reader(csvfile)

for line in reader:
    tmp = [line[0],line[1]]
    print tmp

csvfile.close() 
