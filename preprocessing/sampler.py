__author__ = 'maeglin89273'

import formatter

DEFAULT_SAMPLING_RATE = 5


def sampleChannelsStream(channelData, samplingRate):

    sampledData = [[] if channel else None for channel in channelData]

    for (i, channelValues) in enumerate(zip(*filter(lambda channel: bool(channel), channelData))):
        if i % samplingRate == 0:
            for (j, channelNum) in enumerate((num for num in xrange(len(channelData)) if channelData[num])):
                sampledData[channelNum].append(channelValues[j])

    return sampledData


def sampleChannelsWrap(channelsWrap, samplingRate):
    for category in channelsWrap:
        for fileNum, channelData in enumerate(channelsWrap[category]):
            sampledData = sampleChannelsStream(channelData, samplingRate)
            channelsWrap[category][fileNum] = sampledData

    return channelsWrap

if __name__ == "__main__":
    channelsWrap = formatter.format({"righthand": 4, "sleep": 1}, [1, 2, 3, 4, 8])
    formatter.writeDataWrap(sampleChannelsWrap(channelsWrap, DEFAULT_SAMPLING_RATE), 2)
