import csv
import os
import Filtering

__author__ = 'Maeglin Liao'


RAW_FILES_DIR = "raw_data"
FORMATTED_FILES_DIR = "formatted_data"
IN_SINGLE_FILE = False
LEADING_BIG_WAVE_VALUE_COUNT = 375

def getRawFilePath(category, fileNum):
    return os.path.join(RAW_FILES_DIR, "%s_%s.txt" % (category, fileNum))

def getFormattedFilePath(category, fileNum):
    return os.path.join(FORMATTED_FILES_DIR, "%s_%s.csv" % (category, fileNum))


def skipComments(csvFile):
    for i in xrange(5):
        csvFile.next();

def filterChannels(channels):
    for channel in channels:
        Filtering.filter(channels[channel])

def discardLeadingBigWave(channels):
    for channel in channels:
        values = channels[channel]
        channels[channel] = values[LEADING_BIG_WAVE_VALUE_COUNT:]

    return channels


def readRawData(category, fileNum, extractChannels):
    dataObj = {}
    with open(getRawFilePath(category, fileNum), "rb") as rawDataFile:
                csvFile = csv.reader(rawDataFile)
                skipComments(csvFile);

                dataObj["category"] = category
                dataObj["number"] = fileNum
                channelsDict = {}
                for channel in extractChannels:
                    channelsDict[channel] = []

                for row in csvFile:
                    for extractChannel in extractChannels:
                        #the first column(0) is timestamp
                         channelsDict[extractChannel].append(float(row[extractChannel]))

                filterChannels(channelsDict)
                discardLeadingBigWave(channelsDict)

                dataObj["channels"] = channelsDict

    return dataObj

def format(categories, extractChannels):
    channelsWrap = []
    for (category, fileAmount) in categories.items():
        for fileNum in xrange(1, fileAmount + 1):
                dataObj = readRawData(category, fileNum, extractChannels)
                channelsWrap.append(dataObj)
    return channelsWrap


def writeHeader(csvFile, channelNums):
    channelsCount = len(channelNums)
    headerRows = []
    headerRows.append(["channel_%s" % channelNum for channelNum in channelNums])
    headerRows.append(["float"] * channelsCount)
    headerRows.append([""] * channelsCount)
    csvFile.writerows(headerRows)


def writeChannelSingleFile(dirName, channelNum, channelValues):
    with open(os.path.join(dirName, "channel_%s.csv" % channelNum), "wb") as channelFile:
        csvFile = csv.writer(channelFile, delimiter = ",")
        csvFile.writerow(["channel_value"])
        csvFile.writerow(["float"])
        csvFile.writerow([])

        for i, channelValue in enumerate(channelValues):
           csvFile.writerow([i, channelValue])


def writeChannelsIndividualFiles(channelsDataObj, testingChannel = None):
    dirName = os.path.join(FORMATTED_FILES_DIR, "%s_%s" % (channelsDataObj["category"], channelsDataObj["number"]));
    if (not os.path.exists(dirName)):
        os.makedirs(dirName)

    channels = channelsDataObj["channels"]
    for channelNum in channels:
        writeChannelSingleFile(dirName, channelNum, channels[channelNum])

    if testingChannel:
        writeChannelSingleFile(dirName, "test", channels[testingChannel])

def writeChannelsInSingleFile(channelsDataObj):
    with open(getFormattedFilePath(channelsDataObj["category"], channelsDataObj["number"]), "wb") as formattedDataFile:
        channels = channelsDataObj["channels"]
        csvFile = csv.writer(formattedDataFile, delimiter = ",")
        channelNums = sorted(channels.iterkeys())
        writeHeader(csvFile, channelNums)

        for row in zip(*(channels[channelNum] for channelNum in channelNums)):
            csvFile.writerow(row)

def writeChannelsWrap(channelsWrap, testingChannel = None):
    if IN_SINGLE_FILE:
        for channelsDataObj in channelsWrap:
            writeChannelsInSingleFile(channelsDataObj)
    else:
        for channelsDataObj in channelsWrap:
            writeChannelsIndividualFiles(channelsDataObj, testingChannel)



if __name__ == "__main__":
    channelsWrap = format({"righthand": 4, "sleep": 1}, [1, 2, 3, 4, 8])
    writeChannelsWrap(channelsWrap, 3)
