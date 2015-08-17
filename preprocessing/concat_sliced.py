import csv

TAG = "1"
FILE_COUNT = 21
with open("%s_full.csv" % TAG, "wb") as outFile:
	csvWriter = csv.writer(outFile)
	csvWriter.writerow([i for i in xrange(1, 8 + 1)])

	for i in xrange(1, FILE_COUNT + 1):
		try:
			with open("%s_%s.csv" % (TAG, i), "rb") as inFile:
				csvReader = csv.reader(inFile)
				csvReader.next()
				for row in csvReader:
					csvWriter.writerow(row)
		except:
			print "warning: there is an error on file no.%s" % i
			pass
