import numpy as np
import array
import pywt
__author__ = 'maeglin89273'

class TransformService:


    def fft_transform(self, signal):

        fft_result = np.absolute(np.fft.rfft(signal))
        return fft_result.tolist()


    def dwt_db4_transform(self, signal):
        dwt_result = pywt.wavedec(signal, "db4")
        return np.hstack(dwt_result).tolist()


    def dwt_coif4_transform(self, signal):
        dwt_result = pywt.wavedec(signal, "coif4")
        return np.hstack(dwt_result).tolist()

