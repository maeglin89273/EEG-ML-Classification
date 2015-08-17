import csv
import os

__author__ = 'maeglin89273'

RAW_DATA_DIR = "raw_data"
SLICED_DIR = "step_sliced"
FORMATTED_FILES_DIR = "formatted_data"
FILE_TAG = "1"
FILE_COUNT = 21


def getSlicedFilePath(fileNum):
    return getPathInRawDataDir("%s_%s.csv" % (FILE_TAG, fileNum))

def getChannelDirPath(channelNum):
    return os.path.join(FORMATTED_FILES_DIR, SLICED_DIR, "channel_%s" % channelNum)

def getPathInRawDataDir(fileName):
    return os.path.join(RAW_DATA_DIR, SLICED_DIR, fileName)

def readData():
    data = []
    for i in xrange(1, FILE_COUNT + 1):
        try:
            path = getSlicedFilePath(i)
            with open(path, "rb") as inFile:
                csvReader = csv.reader(inFile)
                csvReader.next()
                channels = [[] for i in xrange(0, 8)]
                for row in csvReader:
                    for (j, value) in enumerate(row):
                        channels[j].append(value)
                data.append(channels)

        except:
            print "warning: there is an error on file " + path
            pass

    return data


def writeIndividualChannelData(structuredData, channelIdx):
    channelDirPath = getChannelDirPath(channelIdx + 1)
    makeSureDirExist(channelDirPath)
    for (i, channels) in enumerate(structuredData):
        with open(os.path.join(channelDirPath, "%s.csv" % i), "wb") as outFile:
            csvFile = csv.writer(outFile)
            csvFile.writerow(["channel_value"])
            csvFile.writerow(["float"])
            csvFile.writerow([])
            for value in channels[channelIdx]:
                csvFile.writerow([value])

def writeData(structuredData):

    for channelIdx in xrange(8):
        writeIndividualChannelData(structuredData, channelIdx)

def makeSureDirExist(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def startExtract():
    structuredData = readData()
    writeData(structuredData)

if __name__ == "__main__":
    startExtract()



