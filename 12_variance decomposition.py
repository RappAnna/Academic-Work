import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt

# Variance Decomposition

# Data preparation
price_ratios["quarter_end"] = pd.to_datetime(price_ratios["quarter_end"]).dt.normalize()
eps_combined["estimate_date"] = pd.to_datetime(eps_combined["estimate_date"]).dt.normalize()
dps_combined["estimate_date"] = pd.to_datetime(dps_combined["estimate_date"]).dt.normalize()

# Merge price ratios with forecasts
div_data = price_ratios[["quarter_end", "log_pd", "log_pe", "pd_ratio", "pe_ratio"]].merge(
    dps_combined[["estimate_date", "div_growth_1yr"]],
    left_on="quarter_end",
    right_on="estimate_date",
    how="inner",
)

earn_data = price_ratios[["quarter_end", "log_pd", "log_pe", "pd_ratio", "pe_ratio"]].merge(
    eps_combined[["estimate_date", "ear_growth_1yr"]],
    left_on="quarter_end",
    right_on="estimate_date",
    how="inner",
)

# Merge return expectations
gh_return_exp_clean = gh_return_exp.copy()
gh_return_exp_clean.index = pd.to_datetime(gh_return_exp_clean.index).normalize()
gh_return_exp_clean = gh_return_exp_clean.reset_index()
gh_return_exp_clean.rename(columns={"index": "date"}, inplace=True)

# Convert to log returns
gh_return_exp_clean["exp_ret_1y_log"] = np.log(1 + gh_return_exp_clean["exp_ret_1y"])
gh_return_exp_clean["exp_ret_10y_annual_log"] = np.log(1 + gh_return_exp_clean["exp_ret_10y"])

# Merge 1-year return expectations
div_data = div_data.merge(
    gh_return_exp_clean[["date", "exp_ret_1y_log"]],
    left_on="estimate_date",
    right_on="date",
    how="left",
)

earn_data = earn_data.merge(
    gh_return_exp_clean[["date", "exp_ret_1y_log"]],
    left_on="estimate_date",
    right_on="date",
    how="left",
)

# Calculate rho and kappa
mean_log_pd = price_ratios["log_pd"].mean()

price_ratios["log_de"] = np.log(price_ratios["dividend_index"] / price_ratios["earnings_index"])
mean_log_de = price_ratios["log_de"].mean()

rho = np.exp(mean_log_pd) / (1 + np.exp(mean_log_pd))

print(f"  Mean log(P/D): {mean_log_pd:.4f}")
print(f"  Mean log(D/E): {mean_log_de:.4f}")
print(f"  ρ (rho): {rho:.4f}")

kappa = -np.log(rho) - (1 - rho) * np.log(1 / rho - 1)
print(f"  κ (kappa) for P/D: {kappa:.4f}")

# kappa-tilde for price-earnings ratio (includes payout ratio term)
kappa_tilde = kappa + (1 - rho) * mean_log_de
print(f"  κ̃ (kappa tilde) for P/E: {kappa_tilde:.4f}")

# Construct future price ratios
div_data["expected_future_pd"] = (
    div_data["log_pd"] - kappa - div_data["div_growth_1yr"] + div_data["exp_ret_1y_log"]
) / rho

earn_data["expected_future_pe"] = (
    earn_data["log_pe"] - kappa_tilde - earn_data["ear_growth_1yr"] + earn_data["exp_ret_1y_log"]
) / rho

# Newey West
def _nw_lags_rule_of_thumb(n: int) -> int:
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    return max(L, 1)

def nw_slope_and_se(y, x, maxlags=None):
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()

    valid = ~(np.isnan(y) | np.isnan(x))
    y, x = y[valid], x[valid]
    n = len(y)

    if n < 5:
        return np.nan, np.nan

    if maxlags is None:
        maxlags = _nw_lags_rule_of_thumb(n)

    X = np.column_stack([np.ones(n), x])
    res = OLS(y, X).fit()

    try:
        rob = res.get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)
        slope = float(rob.params[1])
        se = float(rob.bse[1])
        return slope, se
    except TypeError:
        V = cov_hac(res, nlags=maxlags, use_correction=True)
        slope = float(res.params[1])
        se = float(np.sqrt(V[1, 1]))
        return slope, se

def significance_stars(se, coef):
    if np.isnan(se) or np.isnan(coef) or se == 0:
        return ""
    t_stat = abs(coef / se)
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(t_stat))
    if p_val < 0.01:
        return "***"
    elif p_val < 0.05:
        return "**"
    elif p_val < 0.10:
        return "*"
    return ""

def variance_decomposition(data, price_ratio_col, growth_col, ratio_type="P/D",
                           compute_se=True, n_bootstrap=1000):

    future_col = f"expected_future_{price_ratio_col.split('_')[1]}"
    complete = data.dropna(subset=[price_ratio_col, growth_col, "exp_ret_1y_log", future_col]).copy()

    if len(complete) == 0:
        print(f"\n No complete observations for {ratio_type} decomposition")
        return None

    if "quarter_end" in complete.columns:
        complete = complete.sort_values("quarter_end")

    print(f"\nSample: {complete['quarter_end'].min()} to {complete['quarter_end'].max()}")
    print(f"N = {len(complete)} quarters")

    var_ratio = complete[price_ratio_col].var()
    print(f"\nVariance of log({ratio_type}): {var_ratio:.6f}")
    print(f"Std Dev of log({ratio_type}): {np.sqrt(var_ratio):.4f}")

    cov_growth = complete[[price_ratio_col, growth_col]].cov().iloc[0, 1]
    cov_ret = complete[[price_ratio_col, "exp_ret_1y_log"]].cov().iloc[0, 1]
    cov_future = complete[[price_ratio_col, future_col]].cov().iloc[0, 1]

    CF1 = cov_growth / var_ratio
    DR1 = -cov_ret / var_ratio
    LT = rho * cov_future / var_ratio

    if compute_se:
        b_cf, se_CF1 = nw_slope_and_se(complete[growth_col].values, complete[price_ratio_col].values)

        b_r, se_r = nw_slope_and_se(complete["exp_ret_1y_log"].values, complete[price_ratio_col].values)
        se_DR1 = se_r

        b_lt, se_LT = nw_slope_and_se((rho * complete[future_col]).values, complete[price_ratio_col].values)

    else:
        se_CF1 = se_DR1 = se_LT = np.nan

    # Display results
    print(f"\n1. Cash Flow News (CF₁):")
    print(f"   Coefficient: {CF1:.4f}{significance_stars(se_CF1, CF1)}")
    if not np.isnan(se_CF1):
        print(f"   Std. Error:  ({se_CF1:.4f})")
    print(f"   cov(E*[Δ_{{t+1}}], {price_ratio_col}) = {cov_growth:.6f}")

    print(f"\n2. Discount Rate News (DR₁):")
    print(f"   Coefficient: {DR1:.4f}{significance_stars(se_DR1, DR1)}")
    if not np.isnan(se_DR1):
        print(f"   Std. Error:  ({se_DR1:.4f})")
    print(f"   cov(E*[r_{{t+1}}], {price_ratio_col}) = {cov_ret:.6f}")

    print(f"\n3. Long-Term Component (LT):")
    print(f"   Coefficient: {LT:.4f}{significance_stars(se_LT, LT)}")
    if not np.isnan(se_LT):
        print(f"   Std. Error:  ({se_LT:.4f})")
    print(f"   cov(E*[{price_ratio_col}_{{t+1}}], {price_ratio_col}) = {cov_future:.6f}")
            
    # Verify decomposition sums to 1
    total = CF1 + DR1 + LT
    print(f"\n4. VERIFICATION:")
    print(f"   CF₁ + DR₁ + LT = {total:.4f}")
    print(f"   (Should ≈ 1.00)")

    return {
        "sample_start": complete["quarter_end"].min(),
        "sample_end": complete["quarter_end"].max(),
        "n_obs": len(complete),
        "var_ratio": var_ratio,
        "CF1": CF1,
        "DR1": DR1,
        "LT": LT,
        "total": total,
        "se_CF1": se_CF1,
        "se_DR1": se_DR1,
        "se_LT": se_LT,
        "cov_growth": cov_growth,
        "cov_ret": cov_ret,
        "cov_future": cov_future,
    }

# Full Sample
print("\n" + "="*80)
print("Variance Decomposition: Price-Dividend Ratio")
print("FULL SAMPLE")
print("="*80)

div_results_full = variance_decomposition(
    data=div_data,
    price_ratio_col="log_pd",
    growth_col="div_growth_1yr",
    ratio_type="P/D",
    compute_se=True,
    n_bootstrap=1000, 
)

print("\n" + "="*80)
print("Variance Decomposition: Price-Earnings Ratio")
print("FULL SAMPLE")
print("="*80)

earn_results_full = variance_decomposition(
    data=earn_data,
    price_ratio_col="log_pe",
    growth_col="ear_growth_1yr",
    ratio_type="P/E",
    compute_se=True,
    n_bootstrap=1000,  
)

# Subsample
start_sub = pd.Timestamp("2003-01-01")
end_sub = pd.Timestamp("2015-09-30")

print("\n" + "="*80)
print("Subsample Analysis: 2003Q1 - 2015Q3")
print("="*80)

print("\n" + "="*80)
print("Variance Decomposition: Price-Dividend Ratio")
print("SUBSAMPLE: 2003Q1 - 2015Q3")
print("="*80)

div_data_sub = div_data[(div_data["quarter_end"] >= start_sub) & (div_data["quarter_end"] <= end_sub)].copy()

div_results_sub = variance_decomposition(
    data=div_data_sub,
    price_ratio_col="log_pd",
    growth_col="div_growth_1yr",
    ratio_type="P/D",
    compute_se=True,
    n_bootstrap=1000,  
)

print("\n" + "="*80)
print("Variance Decomposition: Price-Earnings Ratio")
print("SUBSAMPLE: 2003Q1 - 2015Q3")
print("="*80)

earn_data_sub = earn_data[(earn_data["quarter_end"] >= start_sub) & (earn_data["quarter_end"] <= end_sub)].copy()

earn_results_sub = variance_decomposition(
    data=earn_data_sub,
    price_ratio_col="log_pe",
    growth_col="ear_growth_1yr",
    ratio_type="P/E",
    compute_se=True,
    n_bootstrap=1000,  
)



# Extended Variance Decomposition
print("\n" + "="*80)
print("Extended Variance Decomposition")
print("="*80)

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# Newey-West
def _nw_lags_rule_of_thumb(n: int) -> int:
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    return max(L, 1)

def nw_slope_and_se(y, x, maxlags=None):
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()

    valid = ~(np.isnan(y) | np.isnan(x))
    y, x = y[valid], x[valid]
    n = len(y)

    if n < 5:
        return np.nan, np.nan

    if maxlags is None:
        maxlags = _nw_lags_rule_of_thumb(n)

    X = np.column_stack([np.ones(n), x])
    res = OLS(y, X).fit()

    try:
        rob = res.get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)
        slope = float(rob.params[1])
        se = float(rob.bse[1])
        return slope, se
    except TypeError:
        try:
            V = cov_hac(res, nlags=maxlags, use_correction=True)
        except TypeError:
            V = cov_hac(res, nlags=maxlags)
        slope = float(res.params[1])
        se = float(np.sqrt(V[1, 1]))
        return slope, se

# Prepare data
gh_return_exp_clean = gh_return_exp.copy()
gh_return_exp_clean.index = pd.to_datetime(gh_return_exp_clean.index).normalize()
gh_return_exp_clean = gh_return_exp_clean.reset_index()
gh_return_exp_clean.rename(columns={'index': 'date'}, inplace=True)

gh_return_exp_clean['exp_ret_1y_log'] = np.log(1 + gh_return_exp_clean['exp_ret_1y'])
gh_return_exp_clean['exp_ret_10y_annual_log'] = np.log(1 + gh_return_exp_clean['exp_ret_10y'])

div_data_2yr = price_ratios[['quarter_end', 'log_pd', 'log_pe', 'pd_ratio', 'pe_ratio']].merge(
    dps_combined[['estimate_date', 'div_growth_1yr', 'div_growth_2yr']],
    left_on='quarter_end',
    right_on='estimate_date',
    how='inner'
)

earn_data_2yr = price_ratios[['quarter_end', 'log_pd', 'log_pe', 'pd_ratio', 'pe_ratio']].merge(
    eps_combined[['estimate_date', 'ear_growth_1yr', 'ear_growth_2yr']],
    left_on='quarter_end',
    right_on='estimate_date',
    how='inner'
)


div_data_2yr['div_growth_sum_2yr'] = (
    div_data_2yr['div_growth_1yr'] +
    rho * div_data_2yr['div_growth_2yr']
)

earn_data_2yr['ear_growth_sum_2yr'] = (
    earn_data_2yr['ear_growth_1yr'] +
    rho * earn_data_2yr['ear_growth_2yr']
)


div_data_2yr = div_data_2yr.merge(
    gh_return_exp_clean[['date', 'exp_ret_1y_log', 'exp_ret_10y_annual_log']],
    left_on='quarter_end',
    right_on='date',
    how='left'
).drop(columns=['date'])

earn_data_2yr = earn_data_2yr.merge(
    gh_return_exp_clean[['date', 'exp_ret_1y_log', 'exp_ret_10y_annual_log']],
    left_on='quarter_end',
    right_on='date',
    how='left'
).drop(columns=['date'])


def calculate_return_sum_10yr(row):
    ret_1y = row['exp_ret_1y_log']
    ret_10y_annual = row['exp_ret_10y_annual_log']

    if pd.isna(ret_1y) or pd.isna(ret_10y_annual):
        return np.nan

    ret_2to10_avg = (10.0 * ret_10y_annual - ret_1y) / 9.0

    term_1 = ret_1y

    if abs(1 - rho) < 1e-10:
        geometric_sum = 9.0
    else:
        geometric_sum = rho * (1 - rho**9) / (1 - rho)

    term_2_to_10 = ret_2to10_avg * geometric_sum

    return term_1 + term_2_to_10

div_data_2yr['ret_sum_10yr'] = div_data_2yr.apply(calculate_return_sum_10yr, axis=1)
earn_data_2yr['ret_sum_10yr'] = earn_data_2yr.apply(calculate_return_sum_10yr, axis=1)


print("\n" + "="*80)
print("Variance Decomposition: Price-Dividend Ratio (2-YEAR)")
print("FULL SAMPLE")
print("="*80)

div_complete_2yr = div_data_2yr.dropna(subset=['log_pd', 'div_growth_sum_2yr', 'ret_sum_10yr']).copy()
div_complete_2yr = div_complete_2yr.sort_values('quarter_end')

print(f"\nSample: {div_complete_2yr['quarter_end'].min()} to {div_complete_2yr['quarter_end'].max()}")
print(f"N = {len(div_complete_2yr)} quarters")

var_pd_2yr = div_complete_2yr['log_pd'].var()
print(f"\nVariance of log(P/D): {var_pd_2yr:.6f}")

cov_div_2yr = div_complete_2yr[['log_pd', 'div_growth_sum_2yr']].cov().iloc[0, 1]
CF2_pd = cov_div_2yr / var_pd_2yr

cov_ret_10yr = div_complete_2yr[['log_pd', 'ret_sum_10yr']].cov().iloc[0, 1]
DR10_pd = -cov_ret_10yr / var_pd_2yr

_, se_CF2_pd = nw_slope_and_se(div_complete_2yr['div_growth_sum_2yr'].values, div_complete_2yr['log_pd'].values)
_, se_DR10_pd_raw = nw_slope_and_se(div_complete_2yr['ret_sum_10yr'].values, div_complete_2yr['log_pd'].values)
se_DR10_pd = se_DR10_pd_raw

print(f"\n1. Cash Flow News (CF₂):")
print(f"   Coefficient: {CF2_pd:.4f}{significance_stars(se_CF2_pd, CF2_pd)}")
print(f"   Std. Error:  ({se_CF2_pd:.4f})")
print(f"   cov(Σ ρ^{{j-1}}×E*[Δd_{{t+j}}], pd_t) = {cov_div_2yr:.6f}")

print(f"\n2. Discount Rate News (DR₁₀):")
print(f"   Coefficient: {DR10_pd:.4f}{significance_stars(se_DR10_pd, DR10_pd)}")
print(f"   Std. Error:  ({se_DR10_pd:.4f})")
print(f"   cov(Σ ρ^{{j-1}}×E*[r_{{t+j}}], pd_t) = {cov_ret_10yr:.6f}")

print(f"\n3. VERIFICATION:")
print(f"   CF₂ + DR₁₀ = {CF2_pd + DR10_pd:.4f}")


print("\n" + "="*80)
print("VARIANCE DECOMPOSITION: PRICE-EARNINGS RATIO (2-YEAR)")
print("FULL SAMPLE")
print("="*80)

earn_complete_2yr = earn_data_2yr.dropna(subset=['log_pe', 'ear_growth_sum_2yr', 'ret_sum_10yr']).copy()
earn_complete_2yr = earn_complete_2yr.sort_values('quarter_end')

print(f"\nSample: {earn_complete_2yr['quarter_end'].min()} to {earn_complete_2yr['quarter_end'].max()}")
print(f"N = {len(earn_complete_2yr)} quarters")

var_pe_2yr = earn_complete_2yr['log_pe'].var()
print(f"\nVariance of log(P/E): {var_pe_2yr:.6f}")

cov_earn_2yr = earn_complete_2yr[['log_pe', 'ear_growth_sum_2yr']].cov().iloc[0, 1]
CF2_pe = cov_earn_2yr / var_pe_2yr

cov_ret_10yr_pe = earn_complete_2yr[['log_pe', 'ret_sum_10yr']].cov().iloc[0, 1]
DR10_pe = -cov_ret_10yr_pe / var_pe_2yr

_, se_CF2_pe = nw_slope_and_se(earn_complete_2yr['ear_growth_sum_2yr'].values, earn_complete_2yr['log_pe'].values)
_, se_DR10_pe_raw = nw_slope_and_se(earn_complete_2yr['ret_sum_10yr'].values, earn_complete_2yr['log_pe'].values)
se_DR10_pe = se_DR10_pe_raw

print(f"\n1. Cash Flow News (CF₂):")
print(f"   Coefficient: {CF2_pe:.4f}{significance_stars(se_CF2_pe, CF2_pe)}")
print(f"   Std. Error:  ({se_CF2_pe:.4f})")
print(f"   cov(Σ ρ^{{j-1}}×E*[Δe_{{t+j}}], pe_t) = {cov_earn_2yr:.6f}")

print(f"\n2. Discount Rate News (DR₁₀):")
print(f"   Coefficient: {DR10_pe:.4f}{significance_stars(se_DR10_pe, DR10_pe)}")
print(f"   Std. Error:  ({se_DR10_pe:.4f})")
print(f"   cov(Σ ρ^{{j-1}}×E*[r_{{t+j}}], pe_t) = {cov_ret_10yr_pe:.6f}")

print(f"\n3. VERIFICATION:")
print(f"   CF₂ + DR₁₀ = {CF2_pe + DR10_pe:.4f}")


print("\n" + "="*80)
print("Subsample Analysis (2-YEAR): 2003Q1 - 2015Q3")
print("="*80)

# Dividends subsample
div_2yr_2003_2015 = div_data_2yr[
    (div_data_2yr['quarter_end'] >= '2003-01-01') &
    (div_data_2yr['quarter_end'] <= '2015-09-30')
].copy().sort_values('quarter_end')

print("\n" + "="*80)
print("Price-Dividend Ratio (2-Year) - SUBSAMPLE 2003-2015")
print("="*80)

div_complete_2yr_sub = div_2yr_2003_2015.dropna(subset=['log_pd', 'div_growth_sum_2yr', 'ret_sum_10yr']).copy()

print(f"\nSample: {div_complete_2yr_sub['quarter_end'].min()} to {div_complete_2yr_sub['quarter_end'].max()}")
print(f"N = {len(div_complete_2yr_sub)} quarters")

var_pd_2yr_sub = div_complete_2yr_sub['log_pd'].var()
print(f"\nVariance of log(P/D): {var_pd_2yr_sub:.6f}")

cov_div_2yr_sub = div_complete_2yr_sub[['log_pd', 'div_growth_sum_2yr']].cov().iloc[0, 1]
CF2_pd_sub = cov_div_2yr_sub / var_pd_2yr_sub

cov_ret_10yr_sub = div_complete_2yr_sub[['log_pd', 'ret_sum_10yr']].cov().iloc[0, 1]
DR10_pd_sub = -cov_ret_10yr_sub / var_pd_2yr_sub

_, se_CF2_pd_sub = nw_slope_and_se(div_complete_2yr_sub['div_growth_sum_2yr'].values, div_complete_2yr_sub['log_pd'].values)
_, se_DR10_pd_sub_raw = nw_slope_and_se(div_complete_2yr_sub['ret_sum_10yr'].values, div_complete_2yr_sub['log_pd'].values)
se_DR10_pd_sub = se_DR10_pd_sub_raw

print(f"\nCF₂:  {CF2_pd_sub:.4f}{significance_stars(se_CF2_pd_sub, CF2_pd_sub)}")
print(f"      ({se_CF2_pd_sub:.4f})")
print(f"DR₁₀: {DR10_pd_sub:.4f}{significance_stars(se_DR10_pd_sub, DR10_pd_sub)}")
print(f"      ({se_DR10_pd_sub:.4f})")
print(f"Total = {CF2_pd_sub + DR10_pd_sub:.4f}")

earn_2yr_2003_2015 = earn_data_2yr[
    (earn_data_2yr['quarter_end'] >= '2003-01-01') &
    (earn_data_2yr['quarter_end'] <= '2015-09-30')
].copy().sort_values('quarter_end')

print("\n" + "="*80)
print("Price-Earnings Ratio (2-Year) - SUBSAMPLE 2003-2015")
print("="*80)

earn_complete_2yr_sub = earn_2yr_2003_2015.dropna(subset=['log_pe', 'ear_growth_sum_2yr', 'ret_sum_10yr']).copy()

print(f"\nSample: {earn_complete_2yr_sub['quarter_end'].min()} to {earn_complete_2yr_sub['quarter_end'].max()}")
print(f"N = {len(earn_complete_2yr_sub)} quarters")

var_pe_2yr_sub = earn_complete_2yr_sub['log_pe'].var()
print(f"\nVariance of log(P/E): {var_pe_2yr_sub:.6f}")

cov_earn_2yr_sub = earn_complete_2yr_sub[['log_pe', 'ear_growth_sum_2yr']].cov().iloc[0, 1]
CF2_pe_sub = cov_earn_2yr_sub / var_pe_2yr_sub

cov_ret_10yr_pe_sub = earn_complete_2yr_sub[['log_pe', 'ret_sum_10yr']].cov().iloc[0, 1]
DR10_pe_sub = -cov_ret_10yr_pe_sub / var_pe_2yr_sub

_, se_CF2_pe_sub = nw_slope_and_se(earn_complete_2yr_sub['ear_growth_sum_2yr'].values, earn_complete_2yr_sub['log_pe'].values)
_, se_DR10_pe_sub_raw = nw_slope_and_se(earn_complete_2yr_sub['ret_sum_10yr'].values, earn_complete_2yr_sub['log_pe'].values)
se_DR10_pe_sub = se_DR10_pe_sub_raw

print(f"\nCF₂:  {CF2_pe_sub:.4f}{significance_stars(se_CF2_pe_sub, CF2_pe_sub)}")
print(f"      ({se_CF2_pe_sub:.4f})")
print(f"DR₁₀: {DR10_pe_sub:.4f}{significance_stars(se_DR10_pe_sub, DR10_pe_sub)}")
print(f"      ({se_DR10_pe_sub:.4f})")
print(f"Total = {CF2_pe_sub + DR10_pe_sub:.4f}")










# Full Horizon Variance Decomposition
print("=" * 80)
print("Full-Horizon Variance Decomposition")
print("=" * 80)

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.optimize import brentq, minimize

# Newey-West 
def _nw_lags_rule_of_thumb(n: int) -> int:
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    return max(L, 1)


def nw_slope_and_se(y, x, maxlags=None):
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()
    valid = ~(np.isnan(y) | np.isnan(x))
    y, x = y[valid], x[valid]
    n = len(y)
    if n < 5:
        return np.nan, np.nan
    if maxlags is None:
        maxlags = _nw_lags_rule_of_thumb(n)
    X = np.column_stack([np.ones(n), x])
    res = OLS(y, X).fit()
    try:
        rob = res.get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)
        return float(rob.params[1]), float(rob.bse[1])
    except TypeError:
        try:
            V = cov_hac(res, nlags=maxlags, use_correction=True)
        except TypeError:
            V = cov_hac(res, nlags=maxlags)
        return float(res.params[1]), float(np.sqrt(V[1, 1]))


# CF1 and DR1 
def estimate_cf1_dr1_nw(df, price_col, x_cf_col, x_dr_col, date_col=None, nw_lags=None):
    """
    Returns: (CF1, DR1, se_CF1, se_DR1)
      CF1 = slope of x_cf_col on price_col
      DR1 = -slope of x_dr_col on price_col
    SEs are small-sample adjusted Newey-West.
    """
    cols = [price_col, x_cf_col, x_dr_col] + ([date_col] if date_col else [])
    x = df[cols].dropna().copy()
    if date_col:
        x[date_col] = pd.to_datetime(x[date_col])
        x = x.sort_values(date_col)
    n = len(x)
    if n < 20:
        return np.nan, np.nan, np.nan, np.nan
    if nw_lags is None:
        nw_lags = _nw_lags_rule_of_thumb(n)
    b_cf,     se_cf     = nw_slope_and_se(x[x_cf_col].values, x[price_col].values, maxlags=nw_lags)
    b_dr_raw, se_dr_raw = nw_slope_and_se(x[x_dr_col].values, x[price_col].values, maxlags=nw_lags)
    return float(b_cf), float(-b_dr_raw), float(se_cf), float(se_dr_raw)


# Estimate phi from growth horizons with NW SE
def estimate_phi_from_horizons(df, col_1y, col_2y, date_col=None, nw_lags=None):
    x = df[[col_1y, col_2y] + ([date_col] if date_col else [])].dropna().copy()
    if date_col:
        x[date_col] = pd.to_datetime(x[date_col])
        x = x.sort_values(date_col)
    y      = x[col_2y].to_numpy(dtype=float)
    X      = np.column_stack([np.ones(len(x)), x[col_1y].to_numpy(dtype=float)])
    res    = OLS(y, X).fit()
    n      = len(x)
    if nw_lags is None:
        nw_lags = _nw_lags_rule_of_thumb(n)
    try:
        V = cov_hac(res, nlags=nw_lags, use_correction=True)
    except TypeError:
        V = cov_hac(res, nlags=nw_lags)
    alpha  = float(res.params[0])
    phi    = float(res.params[1])
    se_phi = float(np.sqrt(V[1, 1]))
    mu     = alpha / (1 - phi) if abs(1 - phi) > 1e-8 else np.nan
    return phi, se_phi, alpha, mu


# Estimate phi_r from 1y and 10y return expectations 
def estimate_phi_r_with_nw_from_1y_10y(gh_return_data):
    df = gh_return_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "exp_ret_10y" not in df.columns and "exp_ret_10y_annual" in df.columns:
        df["exp_ret_10y"] = df["exp_ret_10y_annual"]
    df = df[["exp_ret_1y", "exp_ret_10y"]].dropna().copy()
    if len(df) < 20:
        return np.nan, np.nan, np.nan, np.nan
    df["r1_log"]  = np.log1p(df["exp_ret_1y"].astype(float))
    df["r10_log"] = np.log1p(df["exp_ret_10y"].astype(float))
    df["x"] = df["r1_log"]
    df["y"] = 10.0 * df["r10_log"] - df["r1_log"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    if len(df) < 20:
        return np.nan, np.nan, np.nan, np.nan

    y_arr   = df["y"].to_numpy(dtype=float)
    X_arr   = np.column_stack([np.ones(len(df)), df["x"].to_numpy(dtype=float)])
    res     = OLS(y_arr, X_arr).fit()
    n       = len(df)
    maxlags = _nw_lags_rule_of_thumb(n)
    try:
        cov = cov_hac(res, nlags=maxlags, use_correction=True)
    except TypeError:
        cov = cov_hac(res, nlags=maxlags)

    c_hat = float(res.params[0])
    b_hat = float(res.params[1])
    se_b  = float(np.sqrt(cov[1, 1]))
    se_c  = float(np.sqrt(cov[0, 0]))

    def A(phi):
        return 9.0 if abs(1 - phi) < 1e-10 else phi * (1 - phi**9) / (1 - phi)

    if not (0 <= b_hat <= 10.0):
        return np.nan, np.nan, np.nan, np.nan
    try:
        phi_hat = brentq(lambda phi: A(phi) - b_hat, 1e-10, 0.999999)
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan

    eps    = 1e-6
    Ap     = (A(min(phi_hat + eps, 0.999999)) - A(max(phi_hat - eps, 1e-10))) / (2 * eps)
    se_phi = se_b / abs(Ap) if abs(Ap) > 1e-12 else np.nan

    denom  = 9.0 - A(phi_hat)
    mu_hat = c_hat / denom if abs(denom) > 1e-8 else np.nan
    se_mu  = np.nan
    if np.isfinite(mu_hat) and abs(denom) > 1e-8:
        dmu_dc = 1.0 / denom
        dmu_db = c_hat / (denom**2) * Ap
        se_mu  = float(np.sqrt((dmu_dc**2) * (se_c**2) + (dmu_db**2) * (se_b**2)))

    return float(phi_hat), float(se_phi), float(mu_hat), float(se_mu)


def calculate_CF_DR_se(CF1, se_CF1, phi, se_phi, rho):
    if any(np.isnan(v) for v in [CF1, phi, se_CF1, se_phi]):
        return np.nan
    denom = 1 - rho * phi
    if abs(denom) < 1e-8:
        return np.nan
    df_dCF1 = 1.0 / denom
    df_dphi = (rho * CF1) / (denom**2)
    return float(np.sqrt((df_dCF1**2) * (se_CF1**2) + (df_dphi**2) * (se_phi**2)))


# Joint MLE 
def joint_mle_paper(phi_d_init, mu_d_init, phi_r_init, mu_r_init,
                    g1, g2, x, y, CF1, DR1, rho):
    g1 = np.asarray(g1, float);  g2 = np.asarray(g2, float)
    x  = np.asarray(x,  float);  y  = np.asarray(y,  float)

    if len(g1) < 20 or len(x) < 20 or not np.isfinite(CF1) or not np.isfinite(DR1):
        return (np.nan,) * 6

    def A(phi):
        return 9.0 if abs(1 - phi) < 1e-10 else phi * (1 - phi**9) / (1 - phi)

    def constraint_eq(params):
        phi_d, mu_d, phi_r, mu_r = params
        dd = 1 - rho * phi_d
        dr = 1 - rho * phi_r
        if abs(dd) < 1e-10 or abs(dr) < 1e-10:
            return 1e6
        return (CF1 / dd) + (DR1 / dr) - 1.0

    def neg_loglike(params):
        phi_d, mu_d, phi_r, mu_r = params
        if not (0.0 <= phi_d < 0.999999) or not (0.0 <= phi_r < 0.999999):
            return 1e12
        dd = 1 - rho * phi_d
        dr = 1 - rho * phi_r
        if abs(dd) < 1e-10 or abs(dr) < 1e-10:
            return 1e12
        vd    = (g2 - mu_d) - phi_d * (g1 - mu_d)
        rss_d = float(np.sum(vd**2))
        Ar    = A(phi_r)
        vr    = (y - 9.0 * mu_r) - Ar * (x - mu_r)
        rss_r = float(np.sum(vr**2))
        nd, nr = len(vd), len(vr)
        eps = 1e-12
        return nd * np.log((rss_d / nd) + eps) + nr * np.log((rss_r / nr) + eps)

    x0     = np.array([phi_d_init, mu_d_init, phi_r_init, mu_r_init], dtype=float)
    bounds = [(0.0001, 0.9999), (None, None), (0.0001, 0.9999), (None, None)]
    cons   = [{"type": "eq", "fun": constraint_eq}]
    res    = minimize(neg_loglike, x0=x0, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "disp": False})

    if not res.success:
        print(f"  Warning: Joint MLE did not converge: {res.message}")

    phi_d_hat, mu_d_hat, phi_r_hat, mu_r_hat = res.x
    CF_hat = CF1 / (1 - rho * phi_d_hat)
    DR_hat = DR1 / (1 - rho * phi_r_hat)

    return float(phi_d_hat), float(mu_d_hat), float(phi_r_hat), float(mu_r_hat), \
           float(CF_hat), float(DR_hat)


def prep_growth_horizons_arrays(df, col_1y, col_2y, date_col):
    tmp = df[[date_col, col_1y, col_2y]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    return tmp[col_1y].to_numpy(float), tmp[col_2y].to_numpy(float)


def prep_returns_arrays(gh_return_data):
    df = gh_return_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "exp_ret_10y" not in df.columns and "exp_ret_10y_annual" in df.columns:
        df["exp_ret_10y"] = df["exp_ret_10y_annual"]
    df = df[["exp_ret_1y", "exp_ret_10y"]].dropna().copy()
    df["r1_log"]  = np.log1p(df["exp_ret_1y"].astype(float))
    df["r10_log"] = np.log1p(df["exp_ret_10y"].astype(float))
    return df["r1_log"].to_numpy(float), (10.0 * df["r10_log"] - df["r1_log"]).to_numpy(float)


def fmt(val, se=None):
    if np.isnan(val):
        return "n/a"
    if se is None or np.isnan(se):
        return f"{val:.2f}"
    return f"{val:.2f}\n({se:.2f})"


# Samples
start_sub = pd.Timestamp("2003-01-01")
end_sub   = pd.Timestamp("2015-09-30")

div_data_sub  = div_data[ (div_data["quarter_end"]  >= start_sub) & (div_data["quarter_end"]  <= end_sub)].copy()
earn_data_sub = earn_data[(earn_data["quarter_end"] >= start_sub) & (earn_data["quarter_end"] <= end_sub)].copy()

dps_combined_sub = dps_combined[(dps_combined["estimate_date"] >= start_sub) & (dps_combined["estimate_date"] <= end_sub)].copy()
eps_combined_sub = eps_combined[(eps_combined["estimate_date"] >= start_sub) & (eps_combined["estimate_date"] <= end_sub)].copy()

gh_return_exp_sub = gh_return_exp.copy()
if not isinstance(gh_return_exp_sub.index, pd.DatetimeIndex):
    gh_return_exp_sub.index = pd.to_datetime(gh_return_exp_sub.index)
gh_return_exp_sub = gh_return_exp_sub.sort_index()
gh_return_exp_sub = gh_return_exp_sub[
    (gh_return_exp_sub.index >= start_sub) & (gh_return_exp_sub.index <= end_sub)
].copy()

CF1_pd_full, DR1_pd_full, se_CF1_pd_full, se_DR1_pd_full = estimate_cf1_dr1_nw(
    div_data,      "log_pd", "div_growth_1yr", "exp_ret_1y_log", date_col="quarter_end")
CF1_pe_full, DR1_pe_full, se_CF1_pe_full, se_DR1_pe_full = estimate_cf1_dr1_nw(
    earn_data,     "log_pe", "ear_growth_1yr", "exp_ret_1y_log", date_col="quarter_end")
CF1_pd_sub,  DR1_pd_sub,  se_CF1_pd_sub,  se_DR1_pd_sub  = estimate_cf1_dr1_nw(
    div_data_sub,  "log_pd", "div_growth_1yr", "exp_ret_1y_log", date_col="quarter_end")
CF1_pe_sub,  DR1_pe_sub,  se_CF1_pe_sub,  se_DR1_pe_sub  = estimate_cf1_dr1_nw(
    earn_data_sub, "log_pe", "ear_growth_1yr", "exp_ret_1y_log", date_col="quarter_end")

g1_d_full,  g2_d_full  = prep_growth_horizons_arrays(dps_combined,     "div_growth_1yr", "div_growth_2yr", "estimate_date")
g1_d_sub,   g2_d_sub   = prep_growth_horizons_arrays(dps_combined_sub, "div_growth_1yr", "div_growth_2yr", "estimate_date")
g1_e_full,  g2_e_full  = prep_growth_horizons_arrays(eps_combined,     "ear_growth_1yr", "ear_growth_2yr", "estimate_date")
g1_e_sub,   g2_e_sub   = prep_growth_horizons_arrays(eps_combined_sub, "ear_growth_1yr", "ear_growth_2yr", "estimate_date")
x_ret_full, y_ret_full = prep_returns_arrays(gh_return_exp)
x_ret_sub,  y_ret_sub  = prep_returns_arrays(gh_return_exp_sub)


print("\n" + "=" * 80)
print("Price-Dividend Ratio — FULL SAMPLE")
print("=" * 80)

phi_d_full, se_phi_d_full, _, mu_d_full = estimate_phi_from_horizons(
    dps_combined, "div_growth_1yr", "div_growth_2yr", date_col="estimate_date")
CF_m1_pd_full    = CF1_pd_full / (1 - rho * phi_d_full)
DR_m1_pd_full    = 1 - CF_m1_pd_full
se_CF_m1_pd_full = calculate_CF_DR_se(CF1_pd_full, se_CF1_pd_full, phi_d_full, se_phi_d_full, rho)
se_DR_m1_pd_full = se_CF_m1_pd_full
print(f"\nRow 1 (Div growth): φ_d={phi_d_full:.4f} ({se_phi_d_full:.4f})  "
      f"CF={CF_m1_pd_full:.3f}{significance_stars(se_CF_m1_pd_full, CF_m1_pd_full)} ({se_CF_m1_pd_full:.3f})  "
      f"DR={DR_m1_pd_full:.3f}{significance_stars(se_DR_m1_pd_full, DR_m1_pd_full)} ({se_DR_m1_pd_full:.3f})")


phi_r_full, se_phi_r_full, mu_r_full, _ = estimate_phi_r_with_nw_from_1y_10y(gh_return_exp)
DR_m2_pd_full    = DR1_pd_full / (1 - rho * phi_r_full)
CF_m2_pd_full    = 1 - DR_m2_pd_full
se_DR_m2_pd_full = calculate_CF_DR_se(DR1_pd_full, se_DR1_pd_full, phi_r_full, se_phi_r_full, rho)
se_CF_m2_pd_full = se_DR_m2_pd_full
print(f"Row 2 (Returns):    φ_r={phi_r_full:.4f} ({se_phi_r_full:.4f})  "
      f"CF={CF_m2_pd_full:.3f}{significance_stars(se_CF_m2_pd_full, CF_m2_pd_full)} ({se_CF_m2_pd_full:.3f})  "
      f"DR={DR_m2_pd_full:.3f}{significance_stars(se_DR_m2_pd_full, DR_m2_pd_full)} ({se_DR_m2_pd_full:.3f})")

phi_d_j_pd_full, _, phi_r_j_pd_full, _, CF_m3_pd_full, DR_m3_pd_full = joint_mle_paper(
    phi_d_full, mu_d_full, phi_r_full, mu_r_full,
    g1_d_full, g2_d_full, x_ret_full, y_ret_full,
    CF1_pd_full, DR1_pd_full, rho)
se_CF_m3_pd_full = calculate_CF_DR_se(CF1_pd_full, se_CF1_pd_full, phi_d_j_pd_full, se_phi_d_full, rho)
se_DR_m3_pd_full = calculate_CF_DR_se(DR1_pd_full, se_DR1_pd_full, phi_r_j_pd_full, se_phi_r_full, rho)
print(f"Row 3 (Joint MLE):  φ_d={phi_d_j_pd_full:.4f} ({se_phi_d_full:.4f})  "
      f"φ_r={phi_r_j_pd_full:.4f} ({se_phi_r_full:.4f})  "
      f"CF={CF_m3_pd_full:.3f}{significance_stars(se_CF_m3_pd_full, CF_m3_pd_full)} ({se_CF_m3_pd_full:.3f})  "
      f"DR={DR_m3_pd_full:.3f}{significance_stars(se_DR_m3_pd_full, DR_m3_pd_full)} ({se_DR_m3_pd_full:.3f})  "
      f"Sum={CF_m3_pd_full+DR_m3_pd_full:.6f}")


print("\n" + "=" * 80)
print("Price-Dividend Ratio — SUBSAMPLE 2003-2015")
print("=" * 80)

phi_d_sub, se_phi_d_sub, _, mu_d_sub = estimate_phi_from_horizons(
    dps_combined_sub, "div_growth_1yr", "div_growth_2yr", date_col="estimate_date")
CF_m1_pd_sub    = CF1_pd_sub / (1 - rho * phi_d_sub)
DR_m1_pd_sub    = 1 - CF_m1_pd_sub
se_CF_m1_pd_sub = calculate_CF_DR_se(CF1_pd_sub, se_CF1_pd_sub, phi_d_sub, se_phi_d_sub, rho)
se_DR_m1_pd_sub = se_CF_m1_pd_sub
print(f"\nRow 1 (Div growth): φ_d={phi_d_sub:.4f} ({se_phi_d_sub:.4f})  "
      f"CF={CF_m1_pd_sub:.3f}{significance_stars(se_CF_m1_pd_sub, CF_m1_pd_sub)} ({se_CF_m1_pd_sub:.3f})  "
      f"DR={DR_m1_pd_sub:.3f}{significance_stars(se_DR_m1_pd_sub, DR_m1_pd_sub)} ({se_DR_m1_pd_sub:.3f})")

phi_r_sub, se_phi_r_sub, mu_r_sub, _ = estimate_phi_r_with_nw_from_1y_10y(gh_return_exp_sub)
DR_m2_pd_sub    = DR1_pd_sub / (1 - rho * phi_r_sub)
CF_m2_pd_sub    = 1 - DR_m2_pd_sub
se_DR_m2_pd_sub = calculate_CF_DR_se(DR1_pd_sub, se_DR1_pd_sub, phi_r_sub, se_phi_r_sub, rho)
se_CF_m2_pd_sub = se_DR_m2_pd_sub
print(f"Row 2 (Returns):    φ_r={phi_r_sub:.4f} ({se_phi_r_sub:.4f})  "
      f"CF={CF_m2_pd_sub:.3f}{significance_stars(se_CF_m2_pd_sub, CF_m2_pd_sub)} ({se_CF_m2_pd_sub:.3f})  "
      f"DR={DR_m2_pd_sub:.3f}{significance_stars(se_DR_m2_pd_sub, DR_m2_pd_sub)} ({se_DR_m2_pd_sub:.3f})")

phi_d_j_pd_sub, _, phi_r_j_pd_sub, _, CF_m3_pd_sub, DR_m3_pd_sub = joint_mle_paper(
    phi_d_sub, mu_d_sub, phi_r_sub, mu_r_sub,
    g1_d_sub, g2_d_sub, x_ret_sub, y_ret_sub,
    CF1_pd_sub, DR1_pd_sub, rho)
se_CF_m3_pd_sub = calculate_CF_DR_se(CF1_pd_sub, se_CF1_pd_sub, phi_d_j_pd_sub, se_phi_d_sub, rho)
se_DR_m3_pd_sub = calculate_CF_DR_se(DR1_pd_sub, se_DR1_pd_sub, phi_r_j_pd_sub, se_phi_r_sub, rho)
print(f"Row 3 (Joint MLE):  φ_d={phi_d_j_pd_sub:.4f} ({se_phi_d_sub:.4f})  "
      f"φ_r={phi_r_j_pd_sub:.4f} ({se_phi_r_sub:.4f})  "
      f"CF={CF_m3_pd_sub:.3f}{significance_stars(se_CF_m3_pd_sub, CF_m3_pd_sub)} ({se_CF_m3_pd_sub:.3f})  "
      f"DR={DR_m3_pd_sub:.3f}{significance_stars(se_DR_m3_pd_sub, DR_m3_pd_sub)} ({se_DR_m3_pd_sub:.3f})  "
      f"Sum={CF_m3_pd_sub+DR_m3_pd_sub:.6f}")

print("\n" + "=" * 80)
print("Price-Earnings Ratio — FULL SAMPLE")
print("=" * 80)

phi_e_full, se_phi_e_full, _, mu_e_full = estimate_phi_from_horizons(
    eps_combined, "ear_growth_1yr", "ear_growth_2yr", date_col="estimate_date")
CF_m1_pe_full    = CF1_pe_full / (1 - rho * phi_e_full)
DR_m1_pe_full    = 1 - CF_m1_pe_full
se_CF_m1_pe_full = calculate_CF_DR_se(CF1_pe_full, se_CF1_pe_full, phi_e_full, se_phi_e_full, rho)
se_DR_m1_pe_full = se_CF_m1_pe_full
print(f"\nRow 1 (Earn growth): φ_e={phi_e_full:.4f} ({se_phi_e_full:.4f})  "
      f"CF={CF_m1_pe_full:.3f}{significance_stars(se_CF_m1_pe_full, CF_m1_pe_full)} ({se_CF_m1_pe_full:.3f})  "
      f"DR={DR_m1_pe_full:.3f}{significance_stars(se_DR_m1_pe_full, DR_m1_pe_full)} ({se_DR_m1_pe_full:.3f})")

# phi_r already estimated from same GH data
DR_m2_pe_full    = DR1_pe_full / (1 - rho * phi_r_full)
CF_m2_pe_full    = 1 - DR_m2_pe_full
se_DR_m2_pe_full = calculate_CF_DR_se(DR1_pe_full, se_DR1_pe_full, phi_r_full, se_phi_r_full, rho)
se_CF_m2_pe_full = se_DR_m2_pe_full
print(f"Row 2 (Returns):     φ_r={phi_r_full:.4f} ({se_phi_r_full:.4f})  "
      f"CF={CF_m2_pe_full:.3f}{significance_stars(se_CF_m2_pe_full, CF_m2_pe_full)} ({se_CF_m2_pe_full:.3f})  "
      f"DR={DR_m2_pe_full:.3f}{significance_stars(se_DR_m2_pe_full, DR_m2_pe_full)} ({se_DR_m2_pe_full:.3f})")

phi_e_j_pe_full, _, phi_r_j_pe_full, _, CF_m3_pe_full, DR_m3_pe_full = joint_mle_paper(
    phi_e_full, mu_e_full, phi_r_full, mu_r_full,
    g1_e_full, g2_e_full, x_ret_full, y_ret_full,
    CF1_pe_full, DR1_pe_full, rho)
se_CF_m3_pe_full = calculate_CF_DR_se(CF1_pe_full, se_CF1_pe_full, phi_e_j_pe_full, se_phi_e_full, rho)
se_DR_m3_pe_full = calculate_CF_DR_se(DR1_pe_full, se_DR1_pe_full, phi_r_j_pe_full, se_phi_r_full, rho)
print(f"Row 3 (Joint MLE):   φ_e={phi_e_j_pe_full:.4f} ({se_phi_e_full:.4f})  "
      f"φ_r={phi_r_j_pe_full:.4f} ({se_phi_r_full:.4f})  "
      f"CF={CF_m3_pe_full:.3f}{significance_stars(se_CF_m3_pe_full, CF_m3_pe_full)} ({se_CF_m3_pe_full:.3f})  "
      f"DR={DR_m3_pe_full:.3f}{significance_stars(se_DR_m3_pe_full, DR_m3_pe_full)} ({se_DR_m3_pe_full:.3f})  "
      f"Sum={CF_m3_pe_full+DR_m3_pe_full:.6f}")

print("\n" + "=" * 80)
print("Price-Earnings Ratio — SUBSAMPLE 2003-2015")
print("=" * 80)

phi_e_sub, se_phi_e_sub, _, mu_e_sub = estimate_phi_from_horizons(
    eps_combined_sub, "ear_growth_1yr", "ear_growth_2yr", date_col="estimate_date")
CF_m1_pe_sub    = CF1_pe_sub / (1 - rho * phi_e_sub)
DR_m1_pe_sub    = 1 - CF_m1_pe_sub
se_CF_m1_pe_sub = calculate_CF_DR_se(CF1_pe_sub, se_CF1_pe_sub, phi_e_sub, se_phi_e_sub, rho)
se_DR_m1_pe_sub = se_CF_m1_pe_sub
print(f"\nRow 1 (Earn growth): φ_e={phi_e_sub:.4f} ({se_phi_e_sub:.4f})  "
      f"CF={CF_m1_pe_sub:.3f}{significance_stars(se_CF_m1_pe_sub, CF_m1_pe_sub)} ({se_CF_m1_pe_sub:.3f})  "
      f"DR={DR_m1_pe_sub:.3f}{significance_stars(se_DR_m1_pe_sub, DR_m1_pe_sub)} ({se_DR_m1_pe_sub:.3f})")

DR_m2_pe_sub    = DR1_pe_sub / (1 - rho * phi_r_sub)
CF_m2_pe_sub    = 1 - DR_m2_pe_sub
se_DR_m2_pe_sub = calculate_CF_DR_se(DR1_pe_sub, se_DR1_pe_sub, phi_r_sub, se_phi_r_sub, rho)
se_CF_m2_pe_sub = se_DR_m2_pe_sub
print(f"Row 2 (Returns):     φ_r={phi_r_sub:.4f} ({se_phi_r_sub:.4f})  "
      f"CF={CF_m2_pe_sub:.3f}{significance_stars(se_CF_m2_pe_sub, CF_m2_pe_sub)} ({se_CF_m2_pe_sub:.3f})  "
      f"DR={DR_m2_pe_sub:.3f}{significance_stars(se_DR_m2_pe_sub, DR_m2_pe_sub)} ({se_DR_m2_pe_sub:.3f})")

phi_e_j_pe_sub, _, phi_r_j_pe_sub, _, CF_m3_pe_sub, DR_m3_pe_sub = joint_mle_paper(
    phi_e_sub, mu_e_sub, phi_r_sub, mu_r_sub,
    g1_e_sub, g2_e_sub, x_ret_sub, y_ret_sub,
    CF1_pe_sub, DR1_pe_sub, rho)
se_CF_m3_pe_sub = calculate_CF_DR_se(CF1_pe_sub, se_CF1_pe_sub, phi_e_j_pe_sub, se_phi_e_sub, rho)
se_DR_m3_pe_sub = calculate_CF_DR_se(DR1_pe_sub, se_DR1_pe_sub, phi_r_j_pe_sub, se_phi_r_sub, rho)
print(f"Row 3 (Joint MLE):   φ_e={phi_e_j_pe_sub:.4f} ({se_phi_e_sub:.4f})  "
      f"φ_r={phi_r_j_pe_sub:.4f} ({se_phi_r_sub:.4f})  "
      f"CF={CF_m3_pe_sub:.3f}{significance_stars(se_CF_m3_pe_sub, CF_m3_pe_sub)} ({se_CF_m3_pe_sub:.3f})  "
      f"DR={DR_m3_pe_sub:.3f}{significance_stars(se_DR_m3_pe_sub, DR_m3_pe_sub)} ({se_DR_m3_pe_sub:.3f})  "
      f"Sum={CF_m3_pe_sub+DR_m3_pe_sub:.6f}")


# Plots
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

mpl.rcParams.update({
    "path.simplify": False,
    "agg.path.chunksize": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "figure.dpi": 150,
})

LEGEND_FONTSIZE_FIG4 = 14.5  
LW_FIG4 = 1.7                 
PNG_DPI = 600
RED  = "firebrick"
BLUE = "steelblue"

def save_crisp(fig, base_name: str, also_png: bool = True):
    """Save vector PDF + optional PNG."""
    fig.savefig(f"{base_name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if also_png:
        fig.savefig(f"{base_name}.png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)

def style_time_axis(ax, df, start_year=2005):
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(
        start_year, int(df['quarter_end'].max().year) + 1, 5
    )])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=11)

def crisp_plot(ax, x, y, *, ls, label, color):
    ax.plot(
        x, y,
        color=color,
        linestyle=ls,
        linewidth=LW_FIG4,
        label=label,
        antialiased=True,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        dash_capstyle="butt",
        dash_joinstyle="miter",
    )


print("\n" + "="*80)
print("Figure 4: Full-Horizon Expectations vs Price Ratios")
print("="*80)

import matplotlib.dates as mdates

phi_d_joint_full     = phi_d_j_pd_full
phi_r_joint_full     = phi_r_j_pd_full      

phi_e_joint_full     = phi_e_j_pe_full
phi_r_joint_pe_full  = phi_r_j_pe_full      

fig4_pd = div_data[['quarter_end', 'log_pd', 'div_growth_1yr', 'exp_ret_1y_log']].dropna().copy()
phi_d = phi_d_joint_full
phi_r = phi_r_joint_full
fig4_pd['full_horizon_div'] = fig4_pd['div_growth_1yr'] / (1 - rho * phi_d)
fig4_pd['full_horizon_ret'] = fig4_pd['exp_ret_1y_log'] / (1 - rho * phi_r)
fig4_pd['log_pd_demeaned'] = fig4_pd['log_pd'] - fig4_pd['log_pd'].mean()
fig4_pd['full_horizon_div_demeaned'] = fig4_pd['full_horizon_div'] - fig4_pd['full_horizon_div'].mean()
fig4_pd['full_horizon_ret_demeaned'] = fig4_pd['full_horizon_ret'] - fig4_pd['full_horizon_ret'].mean()
fig4_pd = fig4_pd.sort_values('quarter_end')

fig4_pe = earn_data[['quarter_end', 'log_pe', 'ear_growth_1yr', 'exp_ret_1y_log']].dropna().copy()
phi_e = phi_e_joint_full
phi_r_pe = phi_r_joint_pe_full
fig4_pe['full_horizon_earn'] = fig4_pe['ear_growth_1yr'] / (1 - rho * phi_e)
fig4_pe['full_horizon_ret'] = fig4_pe['exp_ret_1y_log'] / (1 - rho * phi_r_pe)
fig4_pe['log_pe_demeaned'] = fig4_pe['log_pe'] - fig4_pe['log_pe'].mean()
fig4_pe['full_horizon_earn_demeaned'] = fig4_pe['full_horizon_earn'] - fig4_pe['full_horizon_earn'].mean()
fig4_pe['full_horizon_ret_demeaned'] = fig4_pe['full_horizon_ret'] - fig4_pe['full_horizon_ret'].mean()
fig4_pe = fig4_pe.sort_values('quarter_end')


# Panel A: Price-dividend ratio
fig1, ax1 = plt.subplots(figsize=(7, 5))

crisp_plot(
    ax1, fig4_pd["quarter_end"], fig4_pd["full_horizon_div_demeaned"],
    ls="solid",
    label=r'$\frac{1}{1-\rho\phi_d}E_t^*[\Delta d_{t+1}]$',
    color=RED
)
crisp_plot(
    ax1, fig4_pd["quarter_end"], fig4_pd["full_horizon_ret_demeaned"],
    ls="--",
    label=r'$\frac{1}{1-\rho\phi_r}E_t^*[r_{t+1}]$',
    color=RED
)
crisp_plot(
    ax1, fig4_pd["quarter_end"], fig4_pd["log_pd_demeaned"],
    ls=":",
    label=r"$pd_t$",
    color=RED
)

ax1.axhline(y=0, color=RED, linestyle="-", linewidth=0.6, alpha=0.3)

ax1.legend(
    loc="upper right",
    fontsize=LEGEND_FONTSIZE_FIG4,
    frameon=False,
    handlelength=2.4,
    borderaxespad=0.6
)

ax1.spines["top"].set_visible(True)
ax1.spines["right"].set_visible(True)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(-1.2, 1.5)
style_time_axis(ax1, fig4_pd, start_year=2005)

fig1.tight_layout()
save_crisp(fig1, "figure4a_pd", also_png=True)
plt.show()

# Panel B: Price-earnings ratio 
fig2, ax2 = plt.subplots(figsize=(7, 5))

crisp_plot(
    ax2, fig4_pe["quarter_end"], fig4_pe["full_horizon_earn_demeaned"],
    ls="solid",
    label=r'$\frac{1}{1-\rho\phi_e}E_t^*[\Delta e_{t+1}]$',
    color=BLUE
)
crisp_plot(
    ax2, fig4_pe["quarter_end"], fig4_pe["full_horizon_ret_demeaned"],
    ls="--",
    label=r'$\frac{1}{1-\rho\phi_r}E_t^*[r_{t+1}]$',
    color=BLUE
)
crisp_plot(
    ax2, fig4_pe["quarter_end"], fig4_pe["log_pe_demeaned"],
    ls=":",
    label=r"$pe_t$",
    color=BLUE
)

ax2.axhline(y=0, color=BLUE, linestyle="-", linewidth=0.6, alpha=0.3)

ax2.legend(
    loc="upper right",
    fontsize=LEGEND_FONTSIZE_FIG4,
    frameon=False,
    handlelength=2.4,
    borderaxespad=0.6
)

ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)
ax2.grid(True, alpha=0.2)
ax2.set_ylim(-1.2, 1.5)
style_time_axis(ax2, fig4_pe, start_year=2005)

fig2.tight_layout()
save_crisp(fig2, "figure4b_pe", also_png=True)
plt.show()