import numpy as np
import array
import pywt
__author__ = 'maeglin89273'

class TransformOracle:

    def __init__(self):
        self.dwt_mode = pywt.MODES.sym

    def fft_transform(self, signal):

        fft_result = np.absolute(np.fft.rfft(signal))
        return fft_result.tolist()

    def length_after_fft(self, original_length):
        if original_length % 2 == 0:
            return (original_length // 2) + 1
        else:
            return (original_length + 1) // 2

    def dwt_db4_transform(self, signal):
        dwt_result = pywt.wavedec(signal, "db4", mode=self.dwt_mode)
        return np.hstack(dwt_result).tolist()

    def length_after_dwt_db4(self, original_length):
        mother_wavelet = pywt.Wavelet("db4")
        return self.compute_dwt_length(mother_wavelet, original_length)

    def dwt_coif4_transform(self, signal):

        dwt_result = pywt.wavedec(signal, "coif4", mode=self.dwt_mode)
        return np.hstack(dwt_result).tolist()

    def length_after_dwt_coif4(self, original_length):
        mother_wavelet = pywt.Wavelet("coif4")
        return self.compute_dwt_length(mother_wavelet, original_length)

    def compute_dwt_length(self, mother_wavelet, original_length):
        level = pywt.dwt_max_level(original_length, mother_wavelet.dec_len)
        full_length = 0
        coeff_length = original_length
        for i in range(level):
            coeff_length = pywt.dwt_coeff_len(coeff_length, mother_wavelet.dec_len, self.dwt_mode)
            full_length += coeff_length

        return full_length + coeff_length
