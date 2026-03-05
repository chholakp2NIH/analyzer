import os
import sys

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from scipy import stats

from data import DataContainer

# from neuronol.useful_funcs import mergePDFFiles


analysis_dir = os.path.expanduser("~/analysis")
if analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)
from analyzer.data import ClinicalDataContainer
from analyzer.integratedanalyzer import IntegratedAnalyzer, IntegratedPairedAnalyzer

# Given
group_data_filters = {
    "ALL": {},
    "INP": {
        "Category": "Tx-Seeking",
    },
    "NTS-HD": {
        "Category": "Non Tx-Seeking",
        "AUD_Current": True,
    },
    "HC": {
        "Category": "Non Tx-Seeking",
        "AUD_Current": False,
    },
}
scores = [
    "hyperarousal",
    "aggression",
    "Total_Audit_Score",
    "F1_NE_All",
    "F2_EF_All",
    "F3_IS_All",
]
scores_renamed = [
    "Hyperarousal",
    "Aggression",
    "TotalAUDITScore",
    "NegativeEmotionality",
    "ExecutiveDysfunction",
    "IncentiveSalience",
]
dict_renaming = {u: v for u, v in zip(scores, scores_renamed)}
scatter_pairs = [
    ("TotalAUDITScore", "Hyperarousal"),
]
scatter_pairs_covars = [
    ("TotalAUDITScore", "Hyperarousal"),
]
# cols_covar = ["Age", "Sex"]
cols_covar = []
paired_anal_group_pairs = [("INP", "HC")]
keep_cols_0181 = ["MRN", "Category", "AUD_Current"] + scores
keep_cols_ana = ["MRN", "Age", "Sex"]
data_dir = os.path.expanduser("~/data/cii")
results_dir = os.path.expanduser("~/research/results/cii")
subdir = "hyperarousal/validation-0181subset"

# Read the 0181-ANA data
fpath_0181 = os.path.join(
    data_dir, "hyperarousal/ANA Hyperarousal data ANA 0181 scores.xlsx"
)
data_0181 = ClinicalDataContainer(
    fpath_0181,
    usecols=keep_cols_0181,
    low_memory=False,
    preprocess_opts={
        "drop_incomplete_rows": True,
    },
)
#   Rename columns
data_0181.df = data_0181.df.rename(columns=dict_renaming)

# Read the ANA data
fpath_ana = os.path.join(data_dir, "hyperarousal/ana_hyperarousal.csv")
data_ana = ClinicalDataContainer(
    fpath_ana,
    usecols=keep_cols_ana,
    low_memory=False,
    preprocess_opts={
        "drop_incomplete_rows": True,
    },
)
#   Preprocess data (e.g., transform Sex column to 0/1)
sex_codes, sex_uniques = data_ana.df["Sex"].factorize(sort=True)
data_ana.df["Sex"] = sex_codes
print(data_ana.df)

# Merge ANA+0181 data
data_merged = data_ana.merge_with(data_0181, "MRN")

# print(data_0181.df)
print(data_merged.df)

#   Filter data based on analysis group
data_merged_grp = {}
for grp in group_data_filters:
    data_merged_grp[grp] = data_merged.filter(group_data_filters[grp])

# Create df combining data for paired anal group
for grp1, grp2 in paired_anal_group_pairs:
    df1 = data_merged_grp[grp1].df
    df2 = data_merged_grp[grp2].df
    df1["AnalGroup"] = 0
    df2["AnalGroup"] = 1
    df_paired_anal = pd.concat([df1, df2], axis=0)
    data_paired_anal = DataContainer.from_dataframe(df_paired_anal)

    # Define analyzer obj
    analyzer = IntegratedAnalyzer(
        data_dir,
        results_dir,
        data_paired_anal,
        cols_of_interest=["AnalGroup"] + scores_renamed + cols_covar,
    )

    # Calculate group diff
    analyzer.calculate_group_diff(
        "Hyperarousal",
        "AnalGroup",
        cols_covar=cols_covar,
        method="OLS",
    )

    # Validate against t-test
    t, pval = stats.ttest_ind(df1["Hyperarousal"], df2["Hyperarousal"])
    print(f"\nT-test: t={t:0.3f}; p={pval:0.2E}")

    # Validate against Wilcoxon-test
    res = stats.ranksums(df1["Hyperarousal"], df2["Hyperarousal"])
    print(f"\nWilcoxon-test: p={res.pvalue:0.2E}")
