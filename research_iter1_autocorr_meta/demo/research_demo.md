# Autocorr & Meta

## Summary

Comprehensive methodology reference for bias-corrected lag-1 autocorrelation estimation on short sequences (10-40 observations), random-effects meta-analysis with REML for pooling per-treebank effect sizes, and mixed-effects regression with language-family random intercepts. Provides exact formulas, recovered polynomial coefficients from Arnau & Bono (2001) for r1' correction at n=6/10/20/30, Monte Carlo simulation design for Phase 1 validation (1,020,000 datasets), and complete Python implementations via statsmodels, PyMARE, and nolds. Key finding: power at T<50 is below 80% for |phi|<0.40, making meta-analytic pooling essential.

## Research Findings

## Executive Summary

For bias-corrected lag-1 autocorrelation estimation on short sequences (10-40 tokens), we recommend using THREE estimators in the Phase 1 Monte Carlo validation: (1) standard r1 as baseline (lowest variability), (2) the Kendall/Marriott-Pope analytical correction r1 + (1+3r1)/n, and (3) MLE via statsmodels as gold-standard comparator. The Arnau & Bono (2001) r1' polynomial coefficients have been successfully recovered for n=6,10,20,30 and can be validated and extended in Phase 1. A critical finding is that power at T<50 is below 80% for |phi|<0.40 [5], making meta-analytic pooling across sentences within treebanks ESSENTIAL for statistical adequacy. For meta-analysis, PyMARE with REML is recommended as the primary tool, with statsmodels combine_effects as secondary [7, 8]. For mixed-effects regression, statsmodels MixedLM with language-family random intercepts provides all needed functionality [10].

---

## Section 1: Autocorrelation Estimators

### 1A. Standard Lag-1 Autocorrelation (Yule-Walker r1)

The standard estimator is [1, 5]:

```
r1 = sum_{t=1}^{n-1} (x_t - x_bar)(x_{t+1} - x_bar) / sum_{t=1}^{n} (x_t - x_bar)^2
```

The denominator uses all n terms while the numerator uses n-1 cross-products, which is one source of finite-sample bias [3].

### 1B. Known Bias of r1 for AR(1) Processes

The classical first-order bias approximation from Kendall (1954) and Marriott & Pope (1954) is [1, 2]:

```
E[r1] ~ rho - (1 + 3*rho)/n
Bias(r1) ~ -(1 + 3*rho)/n
```

This bias is ALWAYS negative to first order, pushing r1 toward zero and beyond [1, 2]. For known mean, the simpler formula applies: Bias = -2*rho/n [4].

Shaman & Stine (1988) extended this to AR(p) models, showing the O(1/T) bias is a simple linear combination of the true AR coefficients with no interaction terms [4]. Mudelsee (2001) refined the formula using (n-1)^{-1} instead of n^{-1} and added a second-order term proportional to (n-1)^{-2}, which produces the expected decline to zero bias as rho approaches unity [2]. The Marriott-Pope correction is satisfactory up to rho ~ 0.85 [2].

### 1C. Arnau & Bono (2001) r1' Polynomial Bias Correction

This is a KEY finding of the research: the exact polynomial coefficients from the Psicothema companion paper [6] have been successfully recovered:

- **n=6**: Model = -0.1648 - 0.5643*r1 - 0.0916*r1^2 (degree 2)
- **n=10**: Model = -0.0972 - 0.3760*r1 - 0.0676*r1^2 (degree 2)
- **n=20**: Model = -0.0482 - 0.2028*r1 - 0.0333*r1^2 (degree 2)
- **n=30**: Model = -0.0373 - 0.1360*r1 (degree 1)

The corrected estimator is: **r1' = r1 + FittingModel(r1, n)** [3, 6].

The simulation used 10,000 replications per condition, AR(1) processes with stationary initialization (Y_0 from N(0, sigma^2/(1-rho^2))), and a 30-observation burn-in period [6]. The r1' estimator shows less empirical bias than both r1 and r1+, and lower MSE than r1 [3].

For our target n values of 15, 25, and 40 (not covered by the original paper), Phase 1 should DERIVE new polynomial coefficients via the same Monte Carlo approach, using the recovered coefficients at n=10, 20, 30 as validation checkpoints [3, 6].

### 1D. Alternative Estimators

From the comprehensive comparison by Krone, Albers & Timmerman (2017) [5]:

**OLS estimator**: phi_hat_OLS = sum(y_t - y_bar)(y_{t+1} - y_bar) / sum_{t=1}^{T-1}(y_t - y_bar)^2. Key difference from r1: denominator uses T-1 terms [5].

**C-statistic** (Young 1941): phi_hat_C = r1 + [(y_T - y_bar)^2 - (y_1 - y_bar)^2] / [2*sum(y_t - y_bar)^2]. SE depends only on T: SE_C = sqrt((T-2)/((T-1)(T+1))) [5].

**MLE**: Iterative maximization of the exact Gaussian log-likelihood via BFGS. The exact log-likelihood includes the contribution of the initial observation from its stationary distribution [13]. In Python: `from statsmodels.tsa.arima.model import ARIMA; model = ARIMA(y, order=(1,0,0)); results = model.fit()` [14].

**r1+** (Huitema & McKean 1991): r1+ = r1 + 1/n. Simple additive correction for negative bias with positive autocorrelation [15].

**Bayesian Bsr**: Uses symmetrized reference prior pi_sr(phi) = 1/(2*pi*sqrt(1-phi^2)). Lowest bias overall but practically equivalent to MLE for T >= 25 [5].

**Jackknife**: theta_hat_jack = n*theta_hat - (n-1)*mean(theta_hat_{-i}). Removes O(1/n) bias. CAUTION: standard leave-one-out jackknife assumes independence; for time series data, block jackknife methods are needed [16].

### 1E. Comparative Results (Krone et al. 2017)

Key findings from the simulation study (T=10,25,40,50,100; phi=-0.90 to 0.90; 2,000 replications) [5]:

- **Bias**: Lowest for Bsr, then MLE. r1 has largest bias for positive phi.
- **Variability**: Lowest for r1 (between phi=-0.70 to 0.40). OLS and MLE show increasing SD at high phi.
- **Power at T=25**: All estimators need |phi| >= 0.60 for 80% power. At T=100, |phi| >= 0.30 suffices.
- **Power at T=10**: ALL methods perform poorly.
- **MLE vs Bsr**: Differences negligible for practical purposes.
- **Critical insight**: Modifications of r1 show smaller bias but larger variability -- a fundamental bias-variance tradeoff [5, 17].

### 1F. Dou et al. (2026) Review

The most recent comprehensive review analytically demonstrates that bias correction without additional information ALWAYS induces a bias-variance tradeoff [17]. This is critical context: any corrected estimator must be evaluated on BOTH bias AND RMSE, not bias alone.

**Practical Recommendation**: Since we compute EXCESS autocorrelation (real minus RPL baseline), if bias is similar for both real and RPL sequences of the same length, it cancels in the difference. This makes standard r1 potentially optimal despite its bias, because it has the lowest variability and thus maximizes power to detect real-vs-baseline differences.

---

## Section 2: Monte Carlo Simulation Design (Phase 1)

### Simulation Parameters
- **Sequence lengths**: T in {10, 15, 20, 25, 30, 40} [5]
- **True autocorrelation**: phi in {-0.40 to 0.40, step 0.05} = 17 values
- **Replications**: 10,000 per cell (more than 2,000-5,000 in Krone et al.) [5, 6]
- **Total datasets**: 6 x 17 x 10,000 = 1,020,000

### AR(1) Generation (Python)
```python
def generate_ar1(n, phi, sigma=1.0, rng=None):
    rng = rng or np.random.default_rng()
    epsilon = rng.normal(0, sigma, n)
    x = np.zeros(n)
    x[0] = epsilon[0] / np.sqrt(1 - phi**2)  # stationary init
    for t in range(1, n):
        x[t] = phi * x[t-1] + epsilon[t]
    return x
```
Critical: x[0] MUST be drawn from stationary distribution N(0, sigma^2/(1-phi^2)) [5, 6].

### Performance Metrics
Bias, relative bias, RMSE, 95% CI coverage (target 0.95), power (reject H0:phi=0 at alpha=0.05), and MDE (smallest |phi| with power >= 0.80) per T value.

### Power Reality Check
Krone et al. found that for ALL methods, power < 80% for T < 50 when |phi| < 0.40 [5]. This means detecting autocorrelation in individual sentences is LOW POWER. Meta-analytic pooling across sentences is ESSENTIAL.

---

## Section 3: Random-Effects Meta-Analysis

### 3A. Core Formulas

**Fixed-effects pooled**: theta_hat_FE = sum(w_i * theta_hat_i) / sum(w_i), w_i = 1/v_i [7, 9]

**Cochran's Q**: Q = sum(w_i * (theta_hat_i - theta_hat_FE)^2) [9]

**DerSimonian-Laird tau^2**: tau2_DL = max(0, (Q-(K-1)) / (sum(w_i) - sum(w_i^2)/sum(w_i))) [7, 9]

**REML tau^2** (iterative Fisher scoring) [7, 9]:
```
l_REML(tau2) = -(1/2)[sum(log(v_i+tau2)) + log(sum(1/(v_i+tau2)))
               + sum((theta_i - theta_RE)^2/(v_i+tau2))]
```
Starting value = Hedges estimator; convergence when delta(tau2) < 1e-5; max 100 iterations [7].

**Random-effects pooled**: theta_hat_RE = sum(w*_i * theta_hat_i) / sum(w*_i), w*_i = 1/(v_i + tau2) [7, 9]

**Heterogeneity**: I^2 = max(0, (Q-(K-1))/Q) * 100% [11, 12]

**95% Prediction interval**: theta_hat_RE +/- t_{K-2, 0.975} * sqrt(SE^2 + tau2) [11, 12]

### 3B. Python Implementation (statsmodels)
```python
from statsmodels.stats.meta_analysis import combine_effects
res = combine_effects(effect, variance, method_re="iterated", use_t=True, row_names=names)
print(res.summary_frame())  # eff, sd_eff, ci_low, ci_upp, w_fe, w_re
tau2 = res.tau2
fig = res.plot_forest()
```
method_re options: "iterated" (Paule-Mandel, recommended), "chi2"/"dl" (DerSimonian-Laird) [8].

### 3C. Python Implementation (PyMARE -- recommended for REML)
```python
from pymare import Dataset
from pymare.estimators import VarianceBasedLikelihoodEstimator
dset = Dataset(y=effect_sizes, v=variances)
reml = VarianceBasedLikelihoodEstimator(method="REML")
reml.fit_dataset(dset)
summary_df = reml.summary().to_df()
tau2 = reml.tau2
```
Extracts: estimate, se, z-score, p-value, ci_0.025, ci_0.975 [7].

### 3D. Effect Size Construction
Per treebank t with n_t sentences: theta_hat_t = mean(excess_s), v_t = var(excess_s)/n_t. Alternative with length-weighting: w_s = L_s, theta_hat_t = sum(w_s*e_s)/sum(w_s), v_t = 1/sum(w_s). Recommend computing both as robustness check.

---

## Section 4: Mixed-Effects Regression

### Model Specification
```
excess_autocorr ~ modality + word_order + case_richness
                + modality:case_richness + word_order:case_richness
                + mean_sent_length + mean_tree_depth + mean_branching
                + (1|language_family)
```

### Python Implementation
```python
import statsmodels.formula.api as smf
model = smf.mixedlm(
    "excess_autocorr ~ modality + C(word_order) + case_richness + "
    "modality:case_richness + C(word_order):case_richness + "
    "mean_sent_length + mean_tree_depth + mean_branching",
    data=df, groups=df["language_family"], re_formula="1"
)
result = model.fit(reml=True)
```
ICC = cov_re[0,0] / (cov_re[0,0] + scale) [10, 18].

For crossed random effects, use vc_formula with a constant groups vector [10].

---

## Section 5: Supplementary DFA

```python
import nolds
alpha = nolds.dfa(dd_sequence)  # Returns Hurst parameter
```
Interpretation: alpha < 0.5 = anti-persistent, alpha = 0.5 = uncorrelated, alpha > 0.5 = persistent [19].

**CRITICAL WARNING**: DFA requires ~1000+ data points for reliable estimation [20]. For 40-token sentences, results are EXPLORATORY only. Recommend computing DFA on concatenated DD sequences at the treebank level (thousands of tokens) rather than per-sentence [19, 20].

---

## Section 6: Forest Plots

Three approaches: (1) statsmodels `res.plot_forest()` for quick plots [8]; (2) `forestplot` package (`pip install forestplot`) for publication-quality coefficient plots with grouping and annotations [21]; (3) manual matplotlib for full control including diamond for pooled estimate and I^2/tau^2 annotations.

---

## Section 7: Package Summary

| Package | Install | Provides |
|---------|---------|----------|
| numpy >=1.24 | pip install numpy | AR(1) generation, arrays |
| statsmodels >=0.14 | pip install statsmodels | ARIMA MLE, combine_effects, MixedLM |
| pymare >=0.0.9 | pip install pymare | REML meta-analysis, meta-regression |
| nolds >=0.5.2 | pip install nolds | DFA, Hurst exponent |
| forestplot >=0.4 | pip install forestplot | Publication forest plots |
| scipy >=1.10 | pip install scipy | t-distribution, chi2, optimization |
| pingouin >=0.5 | pip install pingouin | ICC (optional) |

---

## Section 8: Decision Flowchart

Phase 1 produces: (1) bias/RMSE table per estimator per (T,phi) cell, (2) calibrated polynomial coefficients for r1' at all target T values, (3) MDE table, (4) chosen estimator. Selection logic: if bias cancels in the excess measure (real-RPL), use r1 for lowest variability; otherwise use analytical correction or MLE. The Phase 1 MDE calibration replaces the provisional |excess| >= 0.05 threshold.

## Sources

[1] [Note on the bias in the estimation of the serial correlation coefficient of AR(1) processes (Mudelsee 2001)](https://link.springer.com/article/10.1007/s003620100077) — Refined the Marriott-Pope (1954) bias formula with second-order terms; confirmed E[r1] ~ rho - (1+3rho)/n with corrections for rho near 1.

[2] [NOTE ON BIAS IN THE ESTIMATION OF AUTOCORRELATION (Kendall 1954)](https://academic.oup.com/biomet/article-abstract/41/3-4/403/230736) — Original derivation of the first-order bias approximation -(1+3rho)/n for the lag-1 autocorrelation estimator.

[3] [Autocorrelation and Bias in Short Time Series: An Alternative Estimator (Arnau & Bono 2001)](https://link.springer.com/article/10.1023/A:1012223430234) — Proposed r1' polynomial bias correction estimator for n<50; derived polynomial fitting models for n=6,10,20,30 via Monte Carlo simulation.

[4] [The Bias of Autoregressive Coefficient Estimators (Shaman & Stine 1988)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478672) — Derived O(1/T) bias formulas for AR(p) least-squares estimators; bias is linear in true coefficients. For AR(1) with unknown mean: -(1+3*alpha)/T.

[5] [A comparative simulation study of AR(1) estimators in short time series (Krone et al. 2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5227053/) — Compared r1, OLS, C-statistic, MLE, and Bayesian estimators. Bsr has lowest bias, r1 has lowest variability. Power <80% for T<50 when |phi|<0.40.

[6] [A program to calculate the empirical bias in autocorrelation estimators (Arnau & Bono 2002)](https://www.psicothema.com/pii?pii=781) — Companion tool paper providing EXACT polynomial coefficients for r1' correction at n=6,10,20,30 and the MATLAB program implementation.

[7] [Run Estimators on a simulated dataset - PyMARE documentation](https://pymare.readthedocs.io/en/latest/auto_examples/02_meta-analysis/plot_meta-analysis_walkthrough.html) — Complete walkthrough of PyMARE REML meta-analysis including VarianceBasedLikelihoodEstimator, Dataset creation, tau2 extraction, and summary DataFrames.

[8] [Meta-Analysis in statsmodels notebook](https://www.statsmodels.org/stable/examples/notebooks/generated/metaanalysis1.html) — Complete code examples for combine_effects with method_re options, summary_frame access, forest plot generation, and fixed vs random effects comparison.

[9] [Methods to estimate the between-study variance and its uncertainty in meta-analysis (Veroniki et al. 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4950030/) — Comprehensive review of tau2 estimators including DL, REML, ML formulas. REML log-likelihood and Fisher scoring algorithm. REML recommended for continuous data.

[10] [Variance Component Analysis - statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html) — Complete examples for nested and crossed random effects using MixedLM, VCSpec, and vc_formula. Shows how to extract variance components and set up crossed designs.

[11] [How to understand and report heterogeneity in a meta-analysis: I-squared and prediction intervals (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11208730/) — Prediction interval formula PI = M +/- t_{K-2} * sqrt(tau2 + V_M). Explains why I2 alone cannot quantify heterogeneity magnitude.

[12] [Chapter 4 Pooling Effect Sizes - Doing Meta-Analysis in R](https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/pooling-es.html) — Detailed formulas for fixed/random effects models, tau2 estimators, I2, H2, and prediction intervals with worked examples.

[13] [Maximum Likelihood Estimation for AR models](https://econ.nsysu.edu.tw/static/file/133/1133/img/3388/Chapter17_MaximumLikelihoodEstimation0701.pdf) — Exact Gaussian log-likelihood for AR(1) including initial observation from stationary distribution and iterative Newton-Raphson solution.

[14] [statsmodels ARIMA.fit() documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html) — API for fitting AR(1) via exact MLE using state-space/Kalman filter methods. Supports multiple estimation methods including statespace and innovations_mle.

[15] [Arnau & Bono 2001 on ResearchGate](https://www.researchgate.net/publication/226035915_Autocorrelation_and_Bias_in_Short_Time_Series_An_Alternative_Estimator) — Confirmed r1+ formula (r1 + 1/n from Huitema & McKean 1991) and r1' outperformance over both r1 and r1+ for n<50.

[16] [Jackknife resampling - Wikipedia](https://en.wikipedia.org/wiki/Jackknife_resampling) — General jackknife bias correction formula: theta_jack = n*theta - (n-1)*mean(theta_{-i}). Removes O(1/n) bias but assumes independence.

[17] [Unravelling the small sample bias in AR(1) models (Dou et al. 2026)](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bmsp.70038?af=R) — Key finding: bias correction without additional information ALWAYS induces a bias-variance tradeoff. Reviews all available correction methods.

[18] [The Intraclass Correlation Coefficient in Mixed Models](https://www.theanalysisfactor.com/the-intraclass-correlation-coefficient-in-mixed-models/) — ICC = sigma2_between / (sigma2_between + sigma2_within) from mixed model output. ICC > 0.1 suggests multilevel modeling needed.

[19] [Nolds 0.6.2 documentation - nolds.dfa()](https://cschoel.github.io/nolds/nolds.html) — Complete API for DFA: parameters, interpretation (alpha<0.5 anti-persistent, =0.5 uncorrelated, >0.5 persistent), minimum nvals>=4, max nvals<=len/10.

[20] [Unbiased detrended fluctuation analysis: Long-range correlations in very short time series](https://www.sciencedirect.com/science/article/abs/pii/S0378437118303637) — DFA requires ~1000 points for reliable estimation; UDFA achieves acceptable results with 500 points. Series of 40 tokens is far below minimum.

[21] [forestplot PyPI - Python package for publication-ready forest plots](https://pypi.org/project/forestplot/) — Lightweight package (pandas/numpy/matplotlib) for forest plots from DataFrames. Supports grouping, annotations, table format. pip install forestplot.

## Follow-up Questions

- What is the optimal number of RPL (random projection/permutation) baseline samples K needed per sentence for stable excess autocorrelation estimates, and does this interact with sentence length?
- Can the bias-cancellation hypothesis (that r1 bias is similar for real and RPL sequences of the same length, so it cancels in the excess measure) be verified analytically, or must it be tested empirically in Phase 1?
- For treebanks with very heterogeneous sentence lengths (e.g., range 10-200 tokens), should the meta-analysis weight by sentence count, total token count, or inverse-variance, and how does the choice affect the pooled estimate and its sensitivity to length-dependent bias?

---
*Generated by AI Inventor Pipeline*
