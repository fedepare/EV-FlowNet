import csv

ofile = open('counter.csv', "w")
writer = csv.writer(ofile, delimiter=',')
with open('disk.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		writer.writerow([row[0], str(int(row[1]) + 2 * (240 / 2 - int(row[1])) - 1), row[2], row[3]])
ofile.close()
