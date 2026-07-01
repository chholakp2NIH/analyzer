import textwrap
from collections import Counter

import matplotlib.pyplot as plt
from scipy.stats import ranksums, shapiro


def test_normality_shapirowilk(data, verbose=True):
    """
    Test for normality of `data` using Shapiro-Wilk Test.
    """
    _, pval = shapiro(data)
    if pval > 0.05:
        normality = True
    else:
        normality = False
    if verbose:
        print(
            f"  - Shapiro-Wilk: {
            "Normal Distribution" if normality else "Non-Normal Distribution"
        } (p = {pval:0.2E})\n"
        )
    return normality, pval


def test_difference_wilcoxon(vec1, vec2, verbose=True):
    """
    Test for differences between `vec1` and `vec2` using Wilcoxon rank sum test (aka Mann-Whitney U test).
    """
    res = ranksums(vec1, vec2)
    if res.pvalue > 0.05:
        different = False
    else:
        different = True
    if verbose:
        print(
            f"  - Wilcoxon: {
            "Significantly Different" if different else "Not Significantly Different"
        } (p = {res.pvalue:0.2E})\n"
        )
    return different, res.pvalue


def generate_histogram(lst, sort_by=None, ax=None, textwrapping=None):
    """
    Counts the number of occurrences of each unique element in `lst`
    and creates a histogram.
    """
    # Count occurrences
    counts = Counter(lst)

    # Sorting (optional)
    if sort_by is not None and sort_by == "element":
        counts = dict(sorted(counts.items(), key=lambda x: x[0]))
    elif sort_by is not None and sort_by == "frequency":
        counts = dict(sorted(counts.items(), key=lambda x: x[1]))

    # Get labels and values for the plot
    labels = counts.keys()
    values = counts.values()

    # Wrap long labels (if `lst` contains strings) for readability
    if all(isinstance(w, str) for w in lst):
        if textwrapping and textwrapping is not None:
            labels = [textwrap.fill(w, width=40) for w in labels]

    # Plot histogram (bar chart)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values)
    ax.set_xlabel("Frequency")

    #   Add labels at the end of the bar
    ax.set_yticks([])
    for i_lbl, (lbl, val) in enumerate(zip(labels, values)):
        ax.text(val + 0.2, i_lbl, lbl, va="center", ha="left", fontsize=9)
    return ax
