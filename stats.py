import pingouin as pg
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats

from .base import PairedAnalyzer, SingleDataAnalyzer
from .data import DataContainer
from .utils.helpers import test_difference_wilcoxon, test_normality_shapirowilk


class PairedStatisticalAnalyzer(PairedAnalyzer):

    def __init__(
        self,
        data_dir,
        results_dir,
        data1: DataContainer,
        data2: DataContainer,
        cols_of_interest,
        label1="Group-A",
        label2="Group-B",
    ) -> None:
        super().__init__(data_dir, results_dir, data1, data2)
        self.cols_of_interest = cols_of_interest
        self.label1 = label1
        self.label2 = label2

        # Run normality checks on all columns of interest in both data containers
        self.normality = {}
        for col in cols_of_interest:
            print(f"Normality check on {self.label1}/'{col}':")
            self.normality[("data1", col)] = test_normality_shapirowilk(self.df1[col])
            print(f"Normality check on {self.label2}/'{col}':")
            self.normality[("data2", col)] = test_normality_shapirowilk(self.df2[col])

    def test_difference_col(self, col):
        """
        Tests difference between datacontainers `data1` and `data2` along `col`.
        """
        # Check whether the vals in any col are normally distributed
        if sum([v[0] for u, v in self.normality.items() if u[1] == col]) < 2:
            method = "Wilcoxon"
        else:
            method = "t"

        # Compare data containers
        if method == "Wilcoxon":
            # res = stats.ranksums(self.df1[col], self.df2[col])
            print(f"Difference between {self.label1} and {self.label2} along '{col}':")
            difference, pval = test_difference_wilcoxon(self.df1[col], self.df2[col])
        elif method == "t":
            ...
        else:
            raise ValueError(f"The `method={method}` is not recognized.")
        return difference, pval, method


class StatisticalAnalyzer(SingleDataAnalyzer):

    def __init__(
        self, data_dir, results_dir, data_container: DataContainer, cols_of_interest
    ) -> None:
        super().__init__(data_dir, results_dir, data_container)
        self.cols_of_interest = cols_of_interest

        # Run normality checks on all columns of interest
        self.normality = {}
        for col in cols_of_interest:
            print(f"Normality check on '{col}':")
            self.normality[col] = test_normality_shapirowilk(self.df[col])

    def calculate_group_diff(
        self, col_outcome: str, col_grp: str, cols_covar=[], method: str = "OLS"
    ):
        """
        Calculates the group difference (identified by `col_grp`) in `col_outcome`.
        Optionally, provide a list of colums to use as covariates (`cols_covar`)
        in the statistical regression model, specified by `method`.
        """
        formula_design_mat = f"{col_outcome} ~ {col_grp}"
        for col in cols_covar:
            formula_design_mat += f" + {col}"
        print("Design matrix formula:", formula_design_mat)

        # Design matrix
        y, X = dmatrices(formula_design_mat, data=self.df, return_type="dataframe")

        # Describe stat model and run fitting
        if method == "OLS":  # continuous outcome; diff in mean
            res = sm.OLS(y, X).fit()
        elif method == "QuantReg-Median":  # continuous outcome; diff in median
            res = sm.QuantReg(y, X).fit(q=0.5)
        elif method == "Gamma":  # continuous outcome; skewed distribution
            res = sm.GLM(y, X, family=sm.families.Gamma()).fit()
        elif method == "Logit":  # binary outcome
            res = sm.Logit(y, X).fit()
        else:
            raise ValueError("Unknown statistical model specified.")
        print(res.summary())  # Summarize model
        print(f"\n{method}-Regression: p-value = {res.pvalues.iloc[1]:0.2E}")
        # print(res.tvalues, res.pvalues)

    def calculate_corr_cols(self, col1, col2):
        """
        Calculates the correlation coeff. between columns 1 and 2.
        """
        # Check whether the vals in either `col1` or `col2` are normally distributed
        if sum([self.normality[col][0] for col in (col1, col2)]) < 2:
            method = "Spearman"
        else:
            method = "Pearson"

        # Calculate corr coef
        if method == "Spearman":
            corr, pval = stats.spearmanr(self.df[col1], self.df[col2])
        elif method == "Pearson":
            corr, pval = stats.pearsonr(self.df[col1], self.df[col2])
        else:
            raise ValueError(f"The `method={method}` is not recognized.")
        text = f"{method} correlation coefficient between '{col1}' and '{col2}' = {corr:0.2f} (p = {pval:0.2E})"
        print(text)
        return corr, pval, method

    def calculate_partial_corr(
        self,
        col1,
        col2,
        cols_covar: list[str],
        method="pearson",
        alternative="two-sided",
    ):
        """
        Calculates the partial correlation coefficient between columns `col1` and `col2`,
        controlling for `cols_covar`, using `method` method. The alternative hypothesis to
        be used by pingouin.partial_corr is specified by `alternative` (default: "two-sided").
        """
        results = pg.partial_corr(
            data=self.df,
            x=col1,
            y=col2,
            covar=cols_covar,
            method=method,
            alternative=alternative,
        )
        # results = self.df.partial_corr(x=col1, y=col2, covar=cols_covar, method=method, alternative=alternative)
        corr, pval = results[["r", "p-val"]].iloc[0]
        return corr, pval, method
