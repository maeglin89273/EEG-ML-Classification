__author__ = 'maeglin89273'

import Formatter

DEFAULT_SAMPLING_RATE = 5


def sampleChannelsStream(channels, samplingRate):
    channelNums = sorted(channels.keys())
    sampledChannels = {channelNum: [] for channelNum in channelNums}

    for (i, channelValues) in enumerate(zip(*(channels[channelNum] for channelNum in channelNums))):
        if i % samplingRate == 0:
            for (i, channelNum) in enumerate(channelNums):
                sampledChannels[channelNum].append(channelValues[i])

    return sampledChannels


def sampleChannelsWrap(channelsWrap, samplingRate):
    for channelsDataObj in channelsWrap:
        sampledChannels = sampleChannelsStream(channelsDataObj["channels"], samplingRate)
        channelsDataObj["channels"] = sampledChannels

    return channelsWrap


if __name__ == "__main__":
    channelsWrap = Formatter.format({"righthand": 4, "sleep": 1}, [1, 2, 3, 4, 8])
    Formatter.writeChannelsWrap(sampleChannelsWrap(channelsWrap, DEFAULT_SAMPLING_RATE), 2)
