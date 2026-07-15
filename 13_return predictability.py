import pandas as pd
import numpy as np
import wrds
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats

# Return Predictability

# Download S&P 500 Returns from CRSP
db = wrds.Connection()
query = """
    SELECT date, vwretd, sprtrn
    FROM crsp.msi
    WHERE date >= '1981-01-01' AND date <= '2023-12-31'
    ORDER BY date
"""
sp500_crsp = db.raw_sql(query)
db.close()

sp500_crsp['date'] = pd.to_datetime(sp500_crsp['date'])
sp500_crsp['return'] = sp500_crsp['sprtrn']
sp500_crsp['log_return'] = np.log(1 + sp500_crsp['return'])

# Aggregate
sp500_crsp = sp500_crsp.set_index('date').sort_index()
sp500_quarterly = pd.DataFrame()
sp500_quarterly['log_return_1q'] = sp500_crsp['log_return'].resample('QE').sum()
sp500_quarterly = sp500_quarterly.reset_index()
sp500_quarterly.columns = ['quarter_end', 'log_return_1q']

# Construct multi-year returns
def calculate_forward_returns(df, return_col, horizons=[1, 3, 5]):
    result = df.copy()
    for h in horizons:
        quarters = h * 4
        result[f'cum_log_return_{h}y'] = (
            result[return_col].rolling(window=quarters).sum().shift(-quarters)
        )
    return result

sp500_quarterly = calculate_forward_returns(sp500_quarterly, 'log_return_1q')

# Discount factor
try:
    alpha = rho
except NameError:
    try:
        mean_log_pd = price_ratios['log_pd'].mean()
        rho = np.exp(mean_log_pd) / (1 + np.exp(mean_log_pd))
        alpha = rho
    except Exception:
        alpha = 0.9774
        rho = alpha
        
# Discounted cumulative returns
def calculate_discounted_returns(df, alpha, horizons=[3, 5]):
    result = df.copy()
    x = result['log_return_1q']

    def one_year_block_ahead(k):
        shift_q = 4 * (k - 1)
        return x.shift(-shift_q).rolling(4).sum().shift(-4)

    r1 = one_year_block_ahead(1)
    r2 = one_year_block_ahead(2)
    r3 = one_year_block_ahead(3)
    r4 = one_year_block_ahead(4)
    r5 = one_year_block_ahead(5)

    result['discounted_return_3y'] = (alpha**0)*r1 + (alpha**1)*r2 + (alpha**2)*r3
    result['discounted_return_5y'] = (
        (alpha**0)*r1 + (alpha**1)*r2 + (alpha**2)*r3 + (alpha**3)*r4 + (alpha**4)*r5
    )
    return result

sp500_quarterly = calculate_discounted_returns(sp500_quarterly, alpha)

# Merge with earnings expectations
for df in [eps_ltg_agg, eps_combined]:
    if 'estimate_date' in df.columns:
        df['estimate_date'] = pd.to_datetime(df['estimate_date']).dt.normalize()

sp500_quarterly['quarter_end'] = pd.to_datetime(sp500_quarterly['quarter_end']).dt.normalize()

regression_data = sp500_quarterly.copy()

regression_data = regression_data.merge(
    eps_ltg_agg[['estimate_date', 'forecast_ltg']].rename(columns={'forecast_ltg': 'LTG'}),
    left_on='quarter_end', right_on='estimate_date', how='left'
).drop(columns=['estimate_date'])

regression_data = regression_data.merge(
    eps_combined[['estimate_date', 'ear_growth_1yr', 'ear_growth_2yr']],
    left_on='quarter_end', right_on='estimate_date', how='left'
).drop(columns=['estimate_date'])

regression_data['growth_1yr'] = regression_data['ear_growth_1yr']
regression_data['growth_2yr'] = regression_data['ear_growth_2yr']

# Define the two subsamples
sub1 = regression_data[
    (regression_data['quarter_end'] >= pd.Timestamp('1981-12-31')) &
    (regression_data['quarter_end'] <= pd.Timestamp('2015-12-31'))
].copy()

sub2 = regression_data[
    (regression_data['quarter_end'] >= pd.Timestamp('2003-03-31')) &
    (regression_data['quarter_end'] <= pd.Timestamp('2015-09-30'))
].copy()

samples = {
    'SUBSAMPLE (1981–2015)':      sub1,
    'SUBSAMPLE (2003Q1–2015Q3)':  sub2
}

# Standardizing variables
def standardize(x):
    return (x - x.mean()) / x.std()

vars_to_std = ['LTG', 'growth_1yr', 'growth_2yr',
               'cum_log_return_1y', 'discounted_return_3y', 'discounted_return_5y']

for label, df in samples.items():
    for var in vars_to_std:
        if var in df.columns:
            valid = df[var].notna()
            df.loc[valid, f'{var}_std'] = standardize(df.loc[valid, var])

# Regressions
def run_regression_with_nw(y, X, nw_lags, var_names=None):
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        if var_names is None:
            var_names = X.columns.tolist()
        X = X.values

    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y_clean = y[mask]
    X_clean = X[mask]
    n = len(y_clean)
    if n < 10:
        return None

    X_with_const = np.column_stack([np.ones(n), X_clean])
    model = OLS(y_clean, X_with_const)
    results = model.fit()

    try:
        rob = results.get_robustcov_results(
            cov_type="HAC", maxlags=nw_lags, use_correction=True
        )
        coefs  = rob.params[1:]
        se_nw  = rob.bse[1:]
    except TypeError:
        cov_nw = cov_hac(results, nlags=nw_lags, use_correction=True)
        coefs  = results.params[1:]
        se_nw  = np.sqrt(np.diag(cov_nw))[1:]

    t_stats  = coefs / se_nw
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - len(coefs) - 1))

    sig_levels = []
    for p in p_values:
        if   p < 0.01: sig_levels.append('***')
        elif p < 0.05: sig_levels.append('**')
        elif p < 0.10: sig_levels.append('*')
        else:          sig_levels.append('')

    return {
        'n': n, 'coefs': coefs, 'se': se_nw,
        't_stats': t_stats, 'p_values': p_values,
        'sig_levels': sig_levels,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'var_names': var_names
    }

def run_all_regressions(data, sample_label):
    print(f"\n{'='*80}")
    print(f"Regressions — {sample_label}")
    print(f"{'='*80}")

    dep_vars = [
        ('cum_log_return_1y_std',    'rt+1',                 4),
        ('discounted_return_3y_std', 'Σ α^(j-1) rt+j (3yr)', 12),
        ('discounted_return_5y_std', 'Σ α^(j-1) rt+j (5yr)', 20)
    ]

    results = {'panel_a': [], 'panel_b': [], 'panel_c': []}

    panels = [
        ('panel_a', 'LTG_std',        'LTG_t',      'RETURNS AND LTG'),
        ('panel_b', 'growth_1yr_std', '1yr growth', 'RETURNS AND GROWTH FORECAST YEAR 1'),
        ('panel_c', 'growth_2yr_std', '2yr growth', 'RETURNS AND GROWTH FORECAST YEAR 2'),
    ]

    for panel_key, reg_col, reg_label, panel_title in panels:
        print(f"\nPANEL {panel_key[-1].upper()}: {panel_title}")
        print("-"*60)

        for dep_var, dep_label, nw_lags in dep_vars:
            result = run_regression_with_nw(
                data[dep_var], data[[reg_col]], nw_lags, var_names=[reg_label]
            )
            if result:
                results[panel_key].append({
                    'dep_var': dep_label,
                    'coef':    result['coefs'][0],
                    'se':      result['se'][0],
                    'sig':     result['sig_levels'][0],
                    'n':       result['n'],
                    'adj_r2':  result['adj_r_squared'] * 100
                })
                print(f"  {dep_label}: coef={result['coefs'][0]:.4f}{result['sig_levels'][0]} "
                      f"SE=({result['se'][0]:.4f}) N={result['n']} "
                      f"Adj R²={result['adj_r_squared']*100:.1f}%")
    return results


def create_table(results, sample_label):
    def _row(panel_key, reg_label):
        rows = []
        p = results[panel_key]
        if len(p) < 3:
            return rows
        rows.append({
            'Regressor': reg_label,
            'rt+1':   f"{p[0]['coef']:.4f}{p[0]['sig']}\n({p[0]['se']:.4f})",
            '3-year': f"{p[1]['coef']:.4f}{p[1]['sig']}\n({p[1]['se']:.4f})",
            '5-year': f"{p[2]['coef']:.4f}{p[2]['sig']}\n({p[2]['se']:.4f})"
        })
        rows.append({
            'Regressor': 'Adj R² (%)',
            'rt+1':   f"{p[0]['adj_r2']:.0f}",
            '3-year': f"{p[1]['adj_r2']:.0f}",
            '5-year': f"{p[2]['adj_r2']:.0f}"
        })
        rows.append({
            'Regressor': 'N',
            'rt+1':   str(p[0]['n']),
            '3-year': str(p[1]['n']),
            '5-year': str(p[2]['n'])
        })
        return rows

    table_data = []
    table_data += _row('panel_a', 'LTG_t')
    table_data += _row('panel_b', 'growth_1yr')
    table_data += _row('panel_c', 'growth_2yr')

    df = pd.DataFrame(table_data)

    print("\n" + "="*80)
    print(f"Table: Return Predictability — {sample_label}")
    print("="*80)
    print(df.to_string(index=False))
    print("\nNOTES:")
    print("- * p<0.10, ** p<0.05, *** p<0.01")
    print(f"- α = {alpha:.4f}")
    print("="*80)

all_results = {}
for sample_label, data in samples.items():
    res = run_all_regressions(data, sample_label)
    all_results[sample_label] = res

print("\n\n" + "#"*80)
print("# Final Tables")
print("#"*80)

for sample_label in samples:
    create_table(all_results[sample_label], sample_label)

# Price ratios
print("\n" + "="*80)
print("Price Ratios")
print("="*80)

earnings_with_pr = regression_data.merge(
    price_ratios[['quarter_end', 'log_pd', 'log_pe']],
    on='quarter_end', how='left'
)

earnings_pr_samples = {
    'SUBSAMPLE (1981–2015)': earnings_with_pr[
        (earnings_with_pr['quarter_end'] >= pd.Timestamp('1981-12-31')) &
        (earnings_with_pr['quarter_end'] <= pd.Timestamp('2015-12-31'))
    ].copy(),
    'SUBSAMPLE (2003Q1–2015Q3)': earnings_with_pr[
        (earnings_with_pr['quarter_end'] >= pd.Timestamp('2003-03-31')) &
        (earnings_with_pr['quarter_end'] <= pd.Timestamp('2015-09-30'))
    ].copy()
}

vars_pr = ['LTG', 'growth_1yr', 'log_pd', 'log_pe', 'discounted_return_5y']
for label, df in earnings_pr_samples.items():
    for var in vars_pr:
        if var in df.columns:
            valid = df[var].notna()
            df.loc[valid, f'{var}_std'] = standardize(df.loc[valid, var])

nw_lags_5y = 20

for sample_label, data in earnings_pr_samples.items():
    print(f"\n{'='*80}")
    print(f"Part 2 — Univariate Price Ratios — {sample_label}")
    print(f"{'='*80}")

    y = data['discounted_return_5y_std']

    result_pd = run_regression_with_nw(y, data[['log_pd_std']], nw_lags_5y, var_names=['log(P/D)'])
    result_pe = run_regression_with_nw(y, data[['log_pe_std']], nw_lags_5y, var_names=['log(P/E)'])

    if result_pd and result_pe:
        print(f"  log(P/D): coef={result_pd['coefs'][0]:.4f}{result_pd['sig_levels'][0]} "
              f"SE=({result_pd['se'][0]:.4f}) Adj R²={result_pd['adj_r_squared']*100:.1f}%")
        print(f"  log(P/E): coef={result_pe['coefs'][0]:.4f}{result_pe['sig_levels'][0]} "
              f"SE=({result_pe['se'][0]:.4f}) Adj R²={result_pe['adj_r_squared']*100:.1f}%")
    print(f"\n{'='*80}")
    print(f"Part 3 — LTG + Price Ratios — {sample_label}")
    print(f"{'='*80}")

    result_col1 = run_regression_with_nw(
        y, data[['LTG_std', 'log_pd_std']], nw_lags_5y, var_names=['LTG', 'log(P/D)']
    )
    result_col2 = run_regression_with_nw(
        y, data[['LTG_std', 'log_pe_std']], nw_lags_5y, var_names=['LTG', 'log(P/E)']
    )

    if result_col1 and result_col2:
        print(f"  LTG + log(P/D): LTG={result_col1['coefs'][0]:.4f}{result_col1['sig_levels'][0]} "
              f"SE=({result_col1['se'][0]:.4f}) "
              f"P/D={result_col1['coefs'][1]:.4f}{result_col1['sig_levels'][1]} "
              f"SE=({result_col1['se'][1]:.4f}) "
              f"Adj R²={result_col1['adj_r_squared']*100:.1f}%")
        print(f"  LTG + log(P/E): LTG={result_col2['coefs'][0]:.4f}{result_col2['sig_levels'][0]} "
              f"SE=({result_col2['se'][0]:.4f}) "
              f"P/E={result_col2['coefs'][1]:.4f}{result_col2['sig_levels'][1]} "
              f"SE=({result_col2['se'][1]:.4f}) "
              f"Adj R²={result_col2['adj_r_squared']*100:.1f}%")