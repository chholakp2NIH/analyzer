import os

import pandas as pd


class DataContainer:

    df: pd.DataFrame

    def __init__(
        self, fpath, usecols=None, sheet_name=0, low_memory=True, cols_dt=[]
    ) -> None:
        self.fpath = fpath
        self.read_data(
            fpath, usecols=usecols, sheet_name=sheet_name, low_memory=low_memory
        )
        for col in cols_dt:
            self.convert2datetime(col)

    def combine_cols(self, col_combined, cols2combine) -> None:
        """
        Creates a new column `col_combined` by adding the values in
        the list of columns specified in `cols2combine`.
        """
        self.df[col_combined] = self.df[cols2combine].sum(axis=1)

    def convert2datetime(self, col) -> None:
        """
        Convert column `col` to be of type `datetime`.
        """
        self.df[col] = pd.to_datetime(self.df[col])

    def filter(self, filter_matchlist):
        """
        Filter dataframe using column-name:value (or column-name:`lambda`) pairs specified in `filter_matchlist`.
        """
        filtered_df = self.df
        for col, match in filter_matchlist.items():
            if callable(match):
                filtered_df = filtered_df[match(filtered_df[col])]
            else:
                filtered_df = filtered_df[filtered_df[col] == match]
        return DataContainer.from_dataframe(filtered_df)

    def read_data(self, fpath, usecols=None, sheet_name=0, low_memory=True) -> None:
        """
        Generic function to read different kinds of data
        as dataframes.
        """
        _, ext = os.path.splitext(fpath)
        if ext == ".xlsx":
            self.df = pd.read_excel(fpath, usecols=usecols, sheet_name=sheet_name)
        elif ext == ".csv":
            self.df = pd.read_csv(fpath, usecols=usecols, low_memory=low_memory)
        elif ext == ".tsv":
            self.df = pd.read_csv(
                fpath, sep="\t", usecols=usecols, low_memory=low_memory
            )
        else:
            raise ValueError(f"Cannot import data from file with extentsion '{ext}'")
        print(f"Read data with {len(self.df)} entry(s).")
        return None

    def merge_with(self, other, on, how="inner", validate="1:1"):
        merged_df = self.df.merge(other.df, on=on, how=how, validate=validate)
        print(f"Created merged data with {len(merged_df)} entries.")
        return DataContainer.from_dataframe(merged_df)

    @classmethod
    def from_dataframe(cls, df):
        obj = cls.__new__(cls)
        obj.fpath = None
        obj.df = df
        return obj


class ClinicalDataContainer(DataContainer):

    def __init__(
        self,
        fpath,
        usecols=None,
        sheet_name=0,
        low_memory=True,
        cols_dt=[],
        preprocess_opts={},
    ) -> None:
        super().__init__(fpath, usecols, sheet_name, low_memory, cols_dt)
        self.preprocess_df(**preprocess_opts)
        self.assert_no_repeated_mrns()

    def preprocess_df(self, drop_incomplete_rows=None, format_dashed_mrns=None) -> None:
        """
        Preprocess clinical data:
        - drop rows with any missing data
        - remove '-' in MRNs (if present) and ensure its data type to be `int`
        """
        if drop_incomplete_rows and drop_incomplete_rows is not None:
            self.df.dropna(inplace=True)  # drop rows with any missing data
        if format_dashed_mrns and format_dashed_mrns is not None:
            if "MRN" in self.df.columns and (
                self.df["MRN"].dtype == str or self.df["MRN"].dtype == object
            ):
                self.df["MRN"] = self.df["MRN"].str.replace("-", "")
                self.df["MRN"] = self.df["MRN"].astype(int)

    def assert_no_repeated_mrns(self) -> None:
        """
        Runs an assert check to ensure that there are no repeated/duplicated
        MRNs in the given df.
        """
        assert (
            len(self.df[self.df.duplicated(subset=["MRN"])]) == 0
        ), f"Duplicate MRNs in {self.fpath}"
