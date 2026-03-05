from PyEMD import EMD

from .base import SignalDataAnalyzer


class SignalProcessor(SignalDataAnalyzer):

    def perform_emd(self):
        emd = EMD()  # initialize EMD object
        imfs = emd.emd(self.signal.series.to_numpy())
        return imfs
