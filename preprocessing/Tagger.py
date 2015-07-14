import copy
from preprocessing import Sampler

__author__ = 'maeglin89273'

import Formatter

DATA_AMOUNT_PER_SEC = 250
TWO_SECS_DATA_AMOUNT = 2 * DATA_AMOUNT_PER_SEC
THREE_SECS_DATA_AMOUNT = 3 * DATA_AMOUNT_PER_SEC
FIVE_SECS_DATA_AMOUNT = 5 * DATA_AMOUNT_PER_SEC

def tagAction(channels, startTime, endTime, actionInterval, actingDuration):
    doNothing = [[] if channel else None for channel in channels]
    action = copy.deepcopy(doNothing)
    actionDataAmount = int(actingDuration * DATA_AMOUNT_PER_SEC)
    actionIntervalDataAmount = int(actionInterval * DATA_AMOUNT_PER_SEC)
    for dataPtr in xrange(int(startTime * DATA_AMOUNT_PER_SEC) + 1, int(endTime * DATA_AMOUNT_PER_SEC), actionIntervalDataAmount):
        for i, channel in enumerate(channels):
            if channel:
                action[i].extend(channel[dataPtr: dataPtr + actionDataAmount])
                doNothing[i].extend(channel[dataPtr + actionDataAmount: dataPtr + actionIntervalDataAmount])

    return doNothing, action

if __name__ == "__main__":
    channelsWrap = Formatter.format({"step": 1, "wink": 1}, [1, 2, 3, 4])
    winkDoNothing, wink = tagAction(channelsWrap["wink"][0], 10, 101, 5, 1)
    stepDoNothing, leftStep = tagAction(channelsWrap["step"][0], 4.87, 100.87, 5, 0.85)
    stepDoNothing2, rightStep = tagAction(channelsWrap["step"][0], 104.87, 200.87, 5, 0.85)
    channelsWrap.clear()
    channelsWrap["winkdonothing"] = [winkDoNothing]
    channelsWrap["wink"] = [wink]
    channelsWrap["stepdonothing"] = [stepDoNothing]
    channelsWrap["leftstep"] = [leftStep]
    channelsWrap["rightstep"] = [rightStep]
    channelsWrap = Sampler.sampleChannelsWrap(channelsWrap, 2)
    Formatter.writeDataWrap(channelsWrap, 3)