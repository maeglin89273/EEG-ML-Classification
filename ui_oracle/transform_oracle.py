import numpy as np
import pywt
__author__ = 'maeglin89273'

class TransformOracle:

    def __init__(self):
        self.fft_freq_start = 0
        self.fft_freq_end = 0

        self.dwt_mode = pywt.MODES.per
        self.mother_wavelet = "coif4"
        self.wt_level_skip = 0
        self.wt_level_max = 4
        self._wt_transform = self.swt_transform
        self.length_after_wt = self.length_after_swt
        self.wt_max_level = self.swt_max_level

    def fft_transform(self, signal):
        fft_result = np.absolute(np.fft.rfft(signal))
        return (fft_result / len(signal)).tolist()

    def length_after_fft(self, original_length):
        if original_length % 2 == 0:
            return ((original_length // 2) + 1)
        else:
            return ((original_length + 1) // 2)

    def set_singal_size_and_sample_rate(self, window_size, sample_rate):
        self.window_sample_ratio = window_size / sample_rate
        self.update_fft_range()


    def set_fft_freq_range(self, start, end):
        self.fft_freq_start = start
        self.fft_freq_end = end
        self.update_fft_range()

    def update_fft_range(self):
        self.fft_start = self.fft_freq_start * self.window_sample_ratio
        self.fft_end = self.fft_freq_end * self.window_sample_ratio

    def ranged_fft_transform(self, signal):
        fft_result = np.absolute(np.fft.rfft(signal))
        return (fft_result / len(signal))[self.fft_start: self.fft_end + 1]

    def set_wt_level_range(self, min, max):
        self.set_wt_level_min(min)
        self.set_wt_level_max(max)

    def set_wt_level_min(self, min):
        self.wt_level_skip = min - 1

    def set_wt_level_max(self, max):
        self.wt_level_max = max

    def set_wt_type(self, type):
        if type == "discrete":
            self._wt_transform = self.dwt_transform
            self.length_after_wt = self.length_after_dwt
            self.wt_max_level = self.dwt_max_level
        else:
            self._wt_transform = self.swt_transform
            self.length_after_wt = self.length_after_swt
            self.wt_max_level = self.swt_max_level

    def wt_transform(self, signal):
        return self._wt_transform(signal).tolist()

    def set_wt_wavelet(self, wavelet_name):
        self.mother_wavelet = wavelet_name

    def length_after_wt(self, original_length):
        pass

    def wt_max_level(self, original_length):
        pass

    def dwt_max_level(self, original_length):
        mother_wavelet = pywt.Wavelet(self.mother_wavelet)
        return pywt.dwt_max_level(original_length, mother_wavelet.dec_len)


    def dwt_transform(self, signal):
        dwt_result = pywt.wavedec(signal, self.mother_wavelet, mode=self.dwt_mode, level=self.wt_level_max)
        return np.hstack(np.hstack(dwt_result[:-self.wt_level_skip or None]))

    def length_after_dwt(self, original_length):
        mother_wavelet = pywt.Wavelet(self.mother_wavelet)
        return self.compute_dwt_length(mother_wavelet, original_length)

    def compute_dwt_length(self, mother_wavelet, original_length):
        full_length = 0
        coeff_length = original_length

        for i in range(self.wt_level_max):
            coeff_length = pywt.dwt_coeff_len(coeff_length, mother_wavelet.dec_len, self.dwt_mode)
            if i >= self.wt_level_skip:
                full_length += coeff_length

        return full_length + coeff_length

    def swt_max_level(self, original_length):
        return pywt.swt_max_level(original_length)

    def swt_transform(self, signal):
        swt_result = pywt.swt(signal, self.mother_wavelet, self.wt_level_max)
        swt_result = swt_result[:-self.wt_level_skip or None]
        coeff_extracted = [swt_result[0][0]]
        for coeff_pair in swt_result:
            coeff_extracted.append(coeff_pair[1])

        return np.hstack(coeff_extracted)

    def length_after_swt(self, original_length):
         return (self.wt_level_max - self.wt_level_skip + 1) * original_length
