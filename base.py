import os

from .data import DataContainer
from .signals import Signal


class BaseAnalyzer:
    def __init__(self, data_dir, results_dir) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)


class SingleDataAnalyzer(BaseAnalyzer):
    def __init__(self, data_dir, results_dir, data_container: DataContainer) -> None:
        super().__init__(data_dir, results_dir)
        self.fpath_data = data_container.fpath
        self.df = data_container.df


class PairedAnalyzer(BaseAnalyzer):
    def __init__(
        self, data_dir, results_dir, data1: DataContainer, data2: DataContainer
    ) -> None:
        super().__init__(data_dir, results_dir)
        self.fpath_data1 = data1.fpath
        self.fpath_data2 = data2.fpath
        self.df1 = data1.df
        self.df2 = data2.df


class SignalDataAnalyzer(BaseAnalyzer):
    def __init__(self, data_dir, results_dir, signal: Signal) -> None:
        super().__init__(data_dir, results_dir)
        self.signal = signal
