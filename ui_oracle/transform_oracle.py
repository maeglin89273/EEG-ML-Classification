__author__ = 'maeglin89273'

class TransformService:

    def __init__(self):
        self.fft_spectrum = None
        self.dwt_spectrum = None

    def get_FFT_spectrum(self):
        return self.fft_spectrum

    def get_DWT_spectrum(self):
        return self.dwt_spectrum

    def tranform(self, signal):
        pass
