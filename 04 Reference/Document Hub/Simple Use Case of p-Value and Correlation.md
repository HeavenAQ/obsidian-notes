---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-14T14:41:00
Status: In-Progress
Last updated time: 2026-07-14T14:41:00
Last edited by: Heaven Chen
Category:
  - Statistics
  - Data Analysis
  - ML
---

# Simple Use Case of p-Value and Correlation

## 1. Question being answered

For each video-level feature, ask:

> Is the feature associated with the improvement obtained by cropping rather than using the whole frame?

There are two separate quantities to report:

- **Effect size:** the correlation coefficient describes the direction and strength of the association.
- **Evidence against the null:** the p-value measures how surprising the observed coefficient would be if the population correlation were zero.

> [!important]
> A small p-value does **not** measure effect size, prove causation, or give the probability that the null hypothesis is true. Report the coefficient, sample size, confidence interval, and p-value together.

## 2. Build the crop-benefit targets

Scores are first deduplicated per video by averaging duplicate configuration copies. Each benefit is then a difference from the whole-frame score.

```python
import pandas as pd

strategies = ["whole", "region", "hand", "full_body"]
scores = {
    strategy: wc.groupby("vid")[f"score_{strategy}"].mean()
    for strategy in strategies
}

S = pd.DataFrame(scores)  # one averaged score per (video, strategy)
df = feat.merge(S, left_on="vid", right_index=True)

df["d_region"] = df["region"] - df["whole"]
df["d_hand"] = df["hand"] - df["whole"]
df["d_body"] = df["full_body"] - df["whole"]
df["best_crop"] = df[["region", "hand", "full_body"]].max(axis=1)
df["d_best"] = df["best_crop"] - df["whole"]
```

Each `score_*` is the benchmark score

$$
\operatorname{score}=\frac{\operatorname{segF1}+\operatorname{timeAcc}}{2}.
$$

Therefore:

- $d_*>0$: the crop outperformed the whole-frame baseline.
- $d_*<0$: the crop underperformed the baseline.
- `d_best`: improvement from selecting the best available crop **after observing the scores**.

> [!caution] Selection optimism
> Because `d_best` is a row-wise maximum, it is optimistically biased when crop scores contain noise. If `d_best` is meant to represent a deployable crop-selection rule, select the crop on training/validation data and evaluate its benefit on held-out data.

## 3. Pearson and Spearman correlation

### 3.1 Pearson correlation

Pearson's sample correlation measures **linear association**:

$$
r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}.
$$

It ranges from $-1$ to $1$. The sign gives the direction; $|r|$ gives the strength of the linear relationship.

For the two-sided test

$$
H_0:\rho=0 \qquad \text{versus} \qquad H_1:\rho\ne 0,
$$

the classical test statistic is

$$
t=r\sqrt{\frac{n-2}{1-r^2}}, \qquad t\sim t_{n-2}\ \text{under }H_0,
$$

and

$$
p=2P\left(T_{n-2}\ge |t|\right).
$$

This test assumes independent observations and, for exact small-sample inference, a bivariate-normal population. Pearson's $r$ is also sensitive to influential outliers.

### 3.2 Worked example

For `n_classes` versus `d_best`, suppose $r=-0.394$ and $n=43$:

$$
t=-0.394\sqrt{\frac{43-2}{1-(-0.394)^2}}\approx -2.74,
$$

with $41$ degrees of freedom. The two-sided p-value is approximately $0.009$. This is evidence against $H_0:\rho=0$, while $r=-0.394$ describes a moderate negative sample association. It does not establish that the number of classes causes crop benefit to change.

### 3.3 Spearman rank correlation

Spearman's $\rho_s$ is Pearson correlation applied to the ranks:

$$
\rho_s=\operatorname{corr}\!\left(\operatorname{rank}(X),\operatorname{rank}(Y)\right).
$$

Use it for a **monotonic** relationship, ordinal data, skewed values, or reduced sensitivity to extreme magnitudes. Without ties, it can also be written as

$$
\rho_s=1-\frac{6\sum_i d_i^2}{n(n^2-1)},
$$

where $d_i$ is the difference between the two ranks. For small samples or many ties, prefer a permutation p-value rather than relying only on an asymptotic approximation.

### 3.4 Pairwise-complete implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def correlation_table(df, features, target, min_n=8):
    rows = []

    for feature in features:
        pair = df[[feature, target]].dropna()  # pairwise-complete rows
        n = len(pair)

        if n < min_n:
            continue
        if pair[feature].nunique() < 2 or pair[target].nunique() < 2:
            continue

        pearson = pearsonr(pair[feature], pair[target])
        spearman = spearmanr(pair[feature], pair[target])

        rows.append({
            "feature": feature,
            "target": target,
            "n": n,
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic,
            "spearman_p": spearman.pvalue,
        })

    return pd.DataFrame(rows).sort_values("pearson_p")


results = correlation_table(df, FEATS, target="d_best")
```

Pairwise deletion preserves more data, but each feature may have a different $n$ and possibly a different subset of videos. Always report `n` per test and investigate whether missingness is systematic.

## 4. Confidence intervals: quantify uncertainty

### 4.1 Fisher interval for Pearson's $r$

For approximately bivariate-normal data, transform $r$ using

$$
z=\operatorname{arctanh}(r), \qquad SE_z=\frac{1}{\sqrt{n-3}}.
$$

A $(1-\alpha)$ interval on the $z$ scale is

$$
z\pm z_{1-\alpha/2}SE_z,
$$

then transform the endpoints back with $\tanh(\cdot)$.

```python
import numpy as np
from scipy.stats import norm


def pearson_ci(r, n, confidence=0.95):
    if n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    critical = norm.ppf(0.5 + confidence / 2)
    lo, hi = z - critical * se, z + critical * se
    return tuple(np.tanh([lo, hi]))
```

### 4.2 Bootstrap interval

Bootstrap paired rows when normality is doubtful or when estimating uncertainty for Spearman's correlation:

```python
import numpy as np
from scipy.stats import spearmanr


def bootstrap_spearman(x, y, n_boot=10_000, seed=0):
    pair = np.column_stack([x, y])
    pair = pair[~np.isnan(pair).any(axis=1)]
    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n_boot):
        sample = pair[rng.integers(0, len(pair), len(pair))]
        if np.unique(sample[:, 0]).size < 2 or np.unique(sample[:, 1]).size < 2:
            continue
        estimates.append(spearmanr(sample[:, 0], sample[:, 1]).statistic)

    return np.quantile(estimates, [0.025, 0.975])
```

The ordinary bootstrap assumes rows are independent. If multiple rows come from the same subject, scene, or recording session, resample entire clusters instead.

## 5. Permutation test: fewer distributional assumptions

Under $H_0$ of no association and exchangeable independent pairs, shuffle one variable while preserving the other. The finite-sample corrected Monte Carlo p-value is

$$
\hat{p}=\frac{1+\sum_{b=1}^{B}\mathbb{1}(|r_b|\ge |r_{\text{obs}}|)}{B+1}.
$$

```python
import numpy as np
from scipy.stats import pearsonr


def permutation_correlation(x, y, n_perm=20_000, seed=0):
    pair = np.column_stack([x, y])
    pair = pair[~np.isnan(pair).any(axis=1)]
    x, y = pair[:, 0], pair[:, 1]

    observed = pearsonr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)

    for b in range(n_perm):
        null[b] = pearsonr(x, rng.permutation(y)).statistic

    p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
    return observed, p_value
```

For clustered or repeated-measures data, unrestricted row shuffling is invalid; permute within appropriate blocks or use a mixed-effects model.

## 6. Multiple comparisons

Testing many features inflates the chance of false positives. If $m$ independent null hypotheses are each tested at level $\alpha$, the probability of at least one false positive is

$$
1-(1-\alpha)^m.
$$

Two common corrections are:

- **Holm correction:** controls the family-wise error rate; appropriate when even one false positive is costly.
- **Benjamini–Hochberg (BH):** controls the false discovery rate; often preferable for exploratory feature screening.

```python
from statsmodels.stats.multitest import multipletests

results["pearson_p_holm"] = multipletests(
    results["pearson_p"], method="holm"
)[1]

results["pearson_q_bh"] = multipletests(
    results["pearson_p"], method="fdr_bh"
)[1]
```

Define the correction family before looking at results—for example, all features tested against `d_best`, or all feature–target pairs in the analysis.

## 7. Regression and partial association

A raw correlation can be explained by a third variable. Multiple linear regression estimates the association with $x_1$ while holding measured covariates fixed:

$$
y_i=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\cdots+\beta_px_{ip}+\varepsilon_i.
$$

The coefficient test uses

$$
t_j=\frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}.
$$

```python
import statsmodels.api as sm

cols = ["n_classes", "video_length", "baseline_score", "d_best"]
model_df = df[cols].dropna()

X = sm.add_constant(model_df[["n_classes", "video_length", "baseline_score"]])
model = sm.OLS(model_df["d_best"], X).fit(cov_type="HC3")
print(model.summary())
```

`HC3` heteroskedasticity-robust standard errors protect inference against non-constant residual variance, but they do not repair nonlinearity, dependence, omitted confounding, or data leakage. Standardize predictors if coefficients need to be compared on a common scale.

Partial correlation is the correlation between residuals after removing the linear effect of covariates $Z$ from both $X$ and $Y$:

$$
r_{XY\cdot Z}=\operatorname{corr}(e_X,e_Y).
$$

```python
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def partial_corr(data, x, y, covariates):
    clean = data[[x, y, *covariates]].dropna()
    Z = clean[covariates]
    x_resid = clean[x] - LinearRegression().fit(Z, clean[x]).predict(Z)
    y_resid = clean[y] - LinearRegression().fit(Z, clean[y]).predict(Z)
    return pearsonr(x_resid, y_resid)
```

For inferential reporting, regression is usually clearer because it directly supplies coefficients, confidence intervals, diagnostics, and extensibility.

## 8. Nonlinear and robust alternatives

| Approach | Use when | What it captures | Main caution |
| --- | --- | --- | --- |
| Pearson $r$ | Relationship is approximately linear | Linear association | Sensitive to outliers |
| Spearman $\rho_s$ | Relationship is monotonic or variables are ordinal | Rank-monotonic association | Misses non-monotonic patterns |
| Kendall $\tau$ | Small samples, ranks, or many pairwise comparisons | Concordance minus discordance | Different numerical scale from $r$ |
| Robust regression | Outliers strongly influence OLS | Conditional trend with reduced outlier influence | The robustness model must match the contamination |
| Spline/GAM | Scatterplot shows curvature | Smooth nonlinear conditional mean | Control flexibility; validate out of sample |
| Mutual information | General dependence is suspected | Linear and nonlinear dependence | No sign; estimator and tuning affect magnitude |

Always begin with a scatterplot. A coefficient alone can hide curvature, clusters, ceiling effects, and a single influential point.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(
    data=df,
    x="n_classes",
    y="d_best",
    robust=True,
    scatter_kws={"alpha": 0.7},
)
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.show()
```

## 9. Recommended analysis workflow

1. **Define the unit of analysis.** Here it should be one independent video, unless videos are clustered by subject or session.
2. **Construct targets without leakage.** Deduplicate first; do not let test outcomes determine a deployed crop-selection rule.
3. **Plot every feature–target pair.** Check shape, outliers, clusters, and missingness.
4. **Choose the estimand.** Pearson for linear association; Spearman/Kendall for monotonic rank association; regression for adjusted effects.
5. **Report uncertainty.** Include $n$, coefficient, 95% confidence interval, raw p-value, and corrected p/q-value.
6. **Correct for multiplicity.** Use Holm for confirmatory testing or BH for exploratory screening.
7. **Check robustness.** Compare Pearson, Spearman, bootstrap/permutation inference, and sensitivity to influential observations.
8. **Model dependence correctly.** Use cluster bootstrap, cluster-robust errors, or mixed-effects models for repeated subjects/sessions.
9. **Separate exploration from confirmation.** Validate selected findings on held-out data or a preregistered follow-up dataset.

## 10. Compact reporting template

> Across $n=43$ independent videos, `n_classes` was moderately negatively associated with `d_best` (Pearson $r=-0.394$, 95% CI $[L,U]$, two-sided $p=.009$, BH-adjusted $q=Q$). Spearman's $\rho_s=R_s$ provided a rank-based sensitivity check. This observational association does not establish causality.

Replace every placeholder and state exactly which hypotheses were included in the multiple-testing correction.
