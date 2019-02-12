import csv

ofile = open('130.csv', "w")
writer = csv.writer(ofile, delimiter=',')
with open('disk.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		writer.writerow([row[0], row[1], str(int(row[2]) + 2 * (264 / 2 - int(row[2])) - 1), row[3]])
ofile.close()
