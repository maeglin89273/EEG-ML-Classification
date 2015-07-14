__author__ = 'maeglin89273'

FILTER_NOTCH_60HZ = {"b": [9.650809863447347e-001, -2.424683201757643e-001, 1.945391494128786e+000, -2.424683201757643e-001, 9.650809863447347e-001],
                "a": [1.000000000000000e+000, -2.467782611297853e-001, 1.944171784691352e+000, -2.381583792217435e-001, 9.313816821269039e-001]}

FILTER_BANDPASS_1_50HZ = {"b": [2.001387256580675e-001, 0.0, -4.002774513161350e-001, 0.0, 2.001387256580675e-001],
          "a": [1.0, -2.355934631131582e+000, 1.941257088655214e+000, -7.847063755334187e-001, 1.999076052968340e-001]}

FILTER_BANDPASS_15_50HZ = {"b": [1.173510367246093e-001, 0.0, -2.347020734492186e-001, 0.0, 1.173510367246093e-001],
          "a": [1.0, -2.137430180172061e+000, 2.038578008108517e+000, -1.070144399200925e+000, 2.946365275879138e-001]}


def filter(channelData):
    filterIIR(FILTER_NOTCH_60HZ["b"], FILTER_NOTCH_60HZ["a"], channelData)
    # filterIIR(FILTER_BANDPASS_1_50HZ["b"], FILTER_BANDPASS_1_50HZ["a"], channelData)
    filterIIR(FILTER_BANDPASS_15_50HZ["b"], FILTER_BANDPASS_15_50HZ["a"], channelData)

def filterIIR(filtB, filtA, data):
    Nback = len(filtB)
    prevY = [0] * Nback
    prevX = [0] * Nback

  #step through data points
    for i in xrange(len(data)):
    #shift the previous outputs
        for j in xrange(Nback - 1, 0, -1):
            prevY[j] = prevY[j - 1]
            prevX[j] = prevX[j - 1]

        #add in the new point
        prevX[0] = data[i]

        #compute the new data point
        out = 0.0
        for j in xrange(Nback):
            out += filtB[j] * prevX[j]
            if j > 0:
                out -= filtA[j] * prevY[j]

        #save output value
        prevY[0] = out
        data[i] = out
