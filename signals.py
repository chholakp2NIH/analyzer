import os
from collections import abc
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

import numpy as np
import pandas as pd
from pandas._typing import InterpolateOptions


class DuplicatePolicy(Enum):

    ERROR = "error"  # raise if duplicates
    FIRST = "first"  # keep first occurrence
    LAST = "last"  # keep first occurrence
    MEAN = "mean"  # average y for duplicate x


@dataclass
class PreprocessConfig:
    sort_x: bool = False
    duplicates: DuplicatePolicy = DuplicatePolicy.ERROR


class Signal:
    series: pd.Series

    def __init__(
        self,
        y: abc.Iterable,
        x: Optional[abc.Iterable] = None,
        x_as_datetime: bool = False,
        datetime_format: Optional[str] = None,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
        cfg: PreprocessConfig = PreprocessConfig(),
    ) -> None:
        """
        Initialize the Signal object.
        """

        # Convert x to dt if it contains dt data and if conversion is needed
        self.x_as_datetime = x_as_datetime
        if x_as_datetime and x is not None:
            try:
                x = pd.to_datetime([w for w in x], format=datetime_format)
            except Exception as e:
                raise e

        # Store x and y as a series
        y_list = [w for w in y]
        self.series = pd.Series(data=y_list, index=x, name=y_lbl)

        # Store other init args in Signal obj
        self.x_lbl = x_lbl
        self.y_lbl = y_lbl

        # Preprocess series as per given config
        self.preprocess(cfg)

    @classmethod
    def concat(
        cls,
        left,
        right,
        ignore_x: bool = True,
        x_as_datetime: bool = False,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
        cfg: PreprocessConfig = PreprocessConfig(),
    ):
        concatenated_series = pd.concat(
            [left.series, right.series], ignore_index=ignore_x
        )
        return cls(
            concatenated_series,
            x=concatenated_series.index,
            y_lbl=y_lbl,
            x_lbl=x_lbl,
            cfg=cfg,
            x_as_datetime=x_as_datetime,
        )

    @classmethod
    def from_spreadsheet_file(
        cls,
        fpath: str,
        col_y: str,
        col_x: Optional[str] = None,
        x_as_datetime: bool = False,
        datetime_format: Optional[str] = None,
        cfg: PreprocessConfig = PreprocessConfig(),
        regex_y: Optional[str] = None,
    ):
        # Use appropriate func to read spreadsheet based on file ext
        _, file_ext = os.path.splitext(fpath)
        if file_ext == ".xlsx":
            df = pd.read_excel(fpath)
        elif file_ext == ".csv":
            df = pd.read_csv(fpath)
        else:
            raise ValueError(f"Unknown file extension {file_ext} in {fpath}")

        # Assign values to init vars and create new object
        y = df[col_y]
        if regex_y is not None:  # extract y using regex (if provided)
            # y = [float(w) for w in y.str.extract(regex_y).to_numpy()]
            y = y.str.extract(regex_y)[0].astype(float)
        x = df[col_x] if col_x is not None else None
        y_lbl = col_y
        x_lbl = col_x
        return cls(
            y,
            x=x,
            x_as_datetime=x_as_datetime,
            datetime_format=datetime_format,
            y_lbl=y_lbl,
            x_lbl=x_lbl,
            cfg=cfg,
        )

    def preprocess(self, cfg: PreprocessConfig):

        # Sort along x
        if cfg.sort_x:
            self.series.sort_index(inplace=True)

        # Handle duplicates in y
        if cfg.duplicates != DuplicatePolicy.ERROR:
            if (cfg.duplicates == DuplicatePolicy.FIRST) or (
                cfg.duplicates == DuplicatePolicy.LAST
            ):
                mask_duplicated = self.series.index.duplicated(
                    keep=cfg.duplicates.value
                )
                self.series = self.series.loc[~mask_duplicated]
            elif cfg.duplicates == DuplicatePolicy.MEAN:
                self.series = cast(pd.Series, self.series.groupby(level=0).mean())
        else:
            mask_duplicated = self.series.index.duplicated()
            if sum(mask_duplicated) > 0:
                raise ValueError(
                    "Duplicate indices found! Set duplicates policy to 'first', 'last', or 'mean'"
                )

    def interpolate(
        self,
        min_res: Optional[float | str] = 1,
        method: InterpolateOptions = "linear",
        order: Optional[int] = None,
    ):
        """
        Interpolate signal Series on intermittent time points, distributed
        at given `freq`.
        """
        # Create a new index with new times to be interpolated
        if self.x_as_datetime:
            new_index = pd.date_range(
                start=self.series.index.min(), end=self.series.index.max(), freq=min_res
            )
        else:
            new_index = pd.Index(
                np.arange(
                    self.series.index.to_series().iloc[0],
                    self.series.index.to_series().iloc[-1] + min_res,
                    min_res,
                )
            )

        # Reindex the series after adding the new index (pd.Series.reindex)
        self.series = self.series.reindex(new_index)

        # Interpolate the series and replace NaNs (pd.Series.interpolate)
        self.series.interpolate(method=method, inplace=True, order=order)
