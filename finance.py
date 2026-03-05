from datetime import date, datetime
from typing import List, Optional, cast

import pandas as pd
import yfinance as yf

from analyzer.base import BaseAnalyzer


class FinanceDataAnalyzer(BaseAnalyzer):

    df: pd.DataFrame

    def __init__(
        self, data_dir: str, results_dir: str, tickers: List, fpath_data: str
    ) -> None:
        super().__init__(data_dir, results_dir)
        self.tickers = tickers
        self.fpath_data = fpath_data

    def download_data(
        self,
        period: Optional[str] = None,
        interval: str = "1d",
        start: Optional[str | datetime | date] = None,
        end: Optional[str | datetime | date] = None,
    ):
        """
        Download Yahoo data as a dataframe for all tickers (if not already exists).
        """
        self.df = cast(
            pd.DataFrame,
            yf.download(
                self.tickers, period=period, interval=interval, start=start, end=end
            ),
        )
        self.df.columns = ["_".join(col) for col in self.df.columns]
        self.df.to_excel(self.fpath_data)
