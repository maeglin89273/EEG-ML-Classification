import csv
import os
import filtering

__author__ = 'Maeglin Liao'


RAW_FILES_DIR = "raw_data"
FORMATTED_FILES_DIR = "formatted_data"
IN_SINGLE_FILE = False
LEADING_BIG_WAVE_VALUE_COUNT = 600

def getRawFilePath(category, fileNum):
    return os.path.join(RAW_FILES_DIR, "%s_%s.txt" % (category, fileNum))

def getFormattedFilePath(category, fileNum):
    return os.path.join(FORMATTED_FILES_DIR, "%s_%s.csv" % (category, fileNum))


def skipComments(csvFile):
    for i in xrange(5):
        csvFile.next()

def filterChannels(channels):
    for channel in channels[1:]:
        if channel:
            filtering.filter(channel)

def discardLeadingBigWave(channels):
    for (i, channel) in enumerate(channels):
        if channel:
            channels[i] = channel[LEADING_BIG_WAVE_VALUE_COUNT:]

    return channels


def readRawData(category, fileNum, extractChannels):

    channelData = [None for i in xrange(9)]
    with open(getRawFilePath(category, fileNum), "rb") as rawDataFile:
        csvFile = csv.reader(rawDataFile)
        skipComments(csvFile)

        for channel in extractChannels:
            channelData[channel] = []

        for row in csvFile:
            for extractChannel in extractChannels:
                #the first column(0) is timestamp
                 channelData[extractChannel].append(float(row[extractChannel]))

        filterChannels(channelData)
        discardLeadingBigWave(channelData)
    return channelData

def format(categories, extractChannels):
    """
    structure as follow:
    {category:[[timestamp, ch1, ch2], [timestamp, ch1, ch2],...,[timestamp, ch1, ch2]]}

    :param category:
    :param fileNum:
    :param extractChannels:
    :return:
    """
    rawDataWrap = {}
    for (category, fileAmount) in categories.items():
        rawDataWrap[category] = []
        for fileNum in xrange(1, fileAmount + 1):
                channelData = readRawData(category, fileNum, extractChannels)
                rawDataWrap[category].append(channelData)
    return rawDataWrap


def writeHeader(csvFile, channelNums):
    channelsCount = len(channelNums)
    headerRows = []
    headerRows.append(["channel_%s" % channelNum for channelNum in channelNums])
    headerRows.append(["float"] * channelsCount)
    headerRows.append([""] * channelsCount)
    csvFile.writerows(headerRows)


def writeChannelSingleFile(dirName, channelNum, channelValues):
    with open(os.path.join(dirName, "channel_%s.csv" % channelNum), "wb") as channelFile:
        csvFile = csv.writer(channelFile)
        csvFile.writerow(["channel_value"])
        csvFile.writerow(["float"])
        csvFile.writerow([])

        for i, channelValue in enumerate(channelValues):
           csvFile.writerow([channelValue])


def writeChannelsIndividualFiles(category, fileNum, channelData, testingChannel = None):
    dirName = os.path.join(FORMATTED_FILES_DIR, "%s_%s" % (category, fileNum));
    if (not os.path.exists(dirName)):
        os.makedirs(dirName)

    for (channelNum, channel) in enumerate(channelData):
        if channel:
            writeChannelSingleFile(dirName, channelNum, channel)

    if testingChannel:
        writeChannelSingleFile(dirName, "test", channelData[testingChannel])

def writeChannelsInSingleFile(category, fileNum, channelData):
    with open(getFormattedFilePath(category, fileNum), "wb") as formattedDataFile:
        csvFile = csv.writer(formattedDataFile)

        writeHeader(csvFile, [i for i in xrange(len(channelData)) if channelData[i]])

        for row in zip(*filter(lambda channel: bool(channel), channelData)):
            csvFile.writerow(row)

def writeDataWrap(dataWrap, testingChannel = None):
    if IN_SINGLE_FILE:
        for category in dataWrap:
            for (fileNum, channelData) in enumerate(dataWrap[category]):
                writeChannelsInSingleFile(category, fileNum + 1, channelData)
    else:
        for category in dataWrap:
            for (fileNum, channelData) in enumerate(dataWrap[category]):
                writeChannelsIndividualFiles(category, fileNum + 1, channelData, testingChannel)



if __name__ == "__main__":
    dataWrap = format({"step": 1}, [1, 2, 3, 4])

    writeDataWrap(dataWrap, 3)
