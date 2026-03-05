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
