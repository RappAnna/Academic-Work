import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Coibion Gorodnichenko Regression

# Prepare data
eps_1_index = eps_1_index[["estimate_date", "quarter_end", "earnings_index_1yr", "ear_growth_1yr"]].copy()
eps_1_index = eps_1_index.rename(columns={
    "earnings_index_1yr": "earnings_index_1yr_forecast",
    "ear_growth_1yr": "ear_growth_1yr_forecast",
})

eps_2_index = eps_2_index[["estimate_date", "quarter_end", "earnings_index_2yr", "ear_growth_2yr"]].copy()
eps_2_index = eps_2_index.rename(columns={
    "earnings_index_2yr": "earnings_index_2yr_forecast",
    "ear_growth_2yr": "ear_growth_2yr_forecast",
})

for df in [eps_1_index, eps_2_index, all_comp_agg]:
    if "estimate_date" in df.columns:
        df["estimate_date"] = pd.to_datetime(df["estimate_date"]).dt.normalize()
    if "quarter_end" in df.columns:
        df["quarter_end"] = pd.to_datetime(df["quarter_end"]).dt.normalize()

all_comp_agg["quarter_end"] = pd.to_datetime(all_comp_agg["quarter_end"]).dt.normalize()

# Create comprehensive dataframe
eps_all = eps_1_index.copy()
eps_all = eps_all.rename(columns={"estimate_date": "estimate_date_t"})

eps_all = eps_all.merge(
    all_comp_agg[["quarter_end", "ear_gr_yoy"]],
    left_on="estimate_date_t",
    right_on="quarter_end",
    how="left"
).drop(columns=["quarter_end_y"]).rename(columns={
    "quarter_end_x": "quarter_end",
    "ear_gr_yoy": "ear_gr_yoy_realized_t"
})

eps_all["forecast_error_eps"] = eps_all["ear_gr_yoy_realized_t"] - eps_all["ear_growth_1yr_forecast"]

eps_all["estimate_date_t-1"] = eps_all["estimate_date_t"] - pd.DateOffset(years=1)

eps_all = eps_all.merge(
    eps_2_index[["estimate_date", "earnings_index_2yr_forecast", "ear_growth_2yr_forecast"]],
    left_on="estimate_date_t-1",
    right_on="estimate_date",
    how="inner",
    validate="m:1"
).drop(columns=["estimate_date"])

eps_all["forecast_revision_eps"] = (
    eps_all["ear_growth_1yr_forecast"] - eps_all["ear_growth_2yr_forecast"]
)

cols_keep_eps = [
    "quarter_end",
    "estimate_date_t",
    "forecast_error_eps",
    "forecast_revision_eps",
    "ear_gr_yoy_realized_t",
    "ear_growth_1yr_forecast",
    "ear_growth_2yr_forecast",
]
eps_all = eps_all[cols_keep_eps]

def time_shift_diagnostic(df, err_col, rev_col, time_col="estimate_date_t"):
    d = df[[time_col, err_col, rev_col]].dropna().sort_values(time_col).copy()
    d["rev_lag1"]  = d[rev_col].shift(1)
    d["rev_lead1"] = d[rev_col].shift(-1)

time_shift_diagnostic(eps_all, "forecast_error_eps", "forecast_revision_eps")

eps_all = eps_all.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
assert not np.any(np.isinf(eps_all.select_dtypes(include=[np.number]))), "EPS data still contains inf!"

CUTOFF = pd.Timestamp("2018-12-31")


# Preparation of Data for LTG Regressions
eps_ltg_all = eps_ltg_agg.copy()
eps_ltg_all = eps_ltg_all.rename(columns={'forecast_ltg': 'forecast_eps_ltg_t'})

# add lagged LTG_t-1
eps_ltg_all = eps_ltg_all.merge(
    eps_ltg_agg[["estimate_date", "forecast_ltg"]].rename(
        columns={"forecast_ltg": "forecast_eps_ltg_t-1"}),
    left_on="estimate_date_t-1",
    right_on="estimate_date",
    how="left"
)
eps_ltg_all = eps_ltg_all.drop(columns=["estimate_date_y"]).rename(
    columns={"estimate_date_x": "estimate_date"})

# add lagged LTG_t-2
eps_ltg_all = eps_ltg_all.merge(
    eps_ltg_agg[["estimate_date", "forecast_ltg"]].rename(
        columns={"forecast_ltg": "forecast_eps_ltg_t-2"}),
    left_on="estimate_date_t-2",
    right_on="estimate_date",
    how="left"
)
eps_ltg_all = eps_ltg_all.drop(columns=["estimate_date_y"]).rename(
    columns={"estimate_date_x": "estimate_date"})

# add lagged LTG_t-3
eps_ltg_all = eps_ltg_all.merge(
    eps_ltg_agg[["estimate_date", "forecast_ltg"]].rename(
        columns={"forecast_ltg": "forecast_eps_ltg_t-3"}),
    left_on="estimate_date_t-3",
    right_on="estimate_date",
    how="left"
)
eps_ltg_all = eps_ltg_all.drop(columns=["estimate_date_y"]).rename(
    columns={"estimate_date_x": "estimate_date"})

eps_ltg_all = eps_ltg_all.merge(
    all_comp_agg[["quarter_end", "ear_gr_5y_ann"]],
    left_on="estimate_date", right_on="quarter_end", how="left"
).drop(columns=["quarter_end"]).rename(columns={"ear_gr_5y_ann": "eps_growth_5yr_realized"})

eps_ltg_all = eps_ltg_all.merge(
    all_comp_agg[["quarter_end", "ear_gr_4y_ann"]],
    left_on="estimate_date", right_on="quarter_end", how="left"
).drop(columns=["quarter_end"]).rename(columns={"ear_gr_4y_ann": "eps_growth_4yr_realized"})

eps_ltg_all = eps_ltg_all.merge(
    all_comp_agg[["quarter_end", "ear_gr_3y_ann"]],
    left_on="estimate_date", right_on="quarter_end", how="left"
).drop(columns=["quarter_end"]).rename(columns={"ear_gr_3y_ann": "eps_growth_3yr_realized"})

eps_ltg_all["eps_LTG_revision_1"] = (
    eps_ltg_all["forecast_eps_ltg_t"] - eps_ltg_all["forecast_eps_ltg_t-1"])
eps_ltg_all["eps_LTG_revision_2"] = (
    eps_ltg_all["forecast_eps_ltg_t"] - eps_ltg_all["forecast_eps_ltg_t-2"])
eps_ltg_all["eps_LTG_revision_3"] = (
    eps_ltg_all["forecast_eps_ltg_t"] - eps_ltg_all["forecast_eps_ltg_t-3"])

eps_ltg_all["eps_forecast_error_5"] = (
    eps_ltg_all["eps_growth_5yr_realized"] - eps_ltg_all["forecast_eps_ltg_t"])
eps_ltg_all["eps_forecast_error_4"] = (
    eps_ltg_all["eps_growth_4yr_realized"] - eps_ltg_all["forecast_eps_ltg_t"])
eps_ltg_all["eps_forecast_error_3"] = (
    eps_ltg_all["eps_growth_3yr_realized"] - eps_ltg_all["forecast_eps_ltg_t"])


# Coibion-Gorodnichenko Regressions
print("="*80)
print("Coibion Gorodnichenko Regressions")
print("="*80)

def run_cg_regression(data, forecast_error_col, revision_col, lags, horizon_label, revision_label):
    reg_data = data[[forecast_error_col, revision_col]].dropna()

    if len(reg_data) < 10:
        print(f"\nInsufficient data for {horizon_label} with {revision_label}")
        return None

    y = reg_data[forecast_error_col]
    X = sm.add_constant(reg_data[revision_col])

    model = sm.OLS(y, X).fit(cov_type='HAC',
                             cov_kwds={'maxlags': lags, 'use_correction': True})

    return model, len(reg_data)

print("\n" + "="*80)
print("EPS - Coibion-Gorodnichenko Regressions")
print("="*80)

horizons_eps = [
    ('eps_forecast_error_3', 'eps_LTG_revision_1', 12, '(et+3-et)/3 - LTGt', 'LTGt - LTGt-1'),
    ('eps_forecast_error_4', 'eps_LTG_revision_1', 16, '(et+4-et)/4 - LTGt', 'LTGt - LTGt-1'),
    ('eps_forecast_error_5', 'eps_LTG_revision_1', 20, '(et+5-et)/5 - LTGt', 'LTGt - LTGt-1'),
    ('eps_forecast_error_3', 'eps_LTG_revision_2', 12, '(et+3-et)/3 - LTGt', 'LTGt - LTGt-2'),
    ('eps_forecast_error_4', 'eps_LTG_revision_2', 16, '(et+4-et)/4 - LTGt', 'LTGt - LTGt-2'),
    ('eps_forecast_error_5', 'eps_LTG_revision_2', 20, '(et+5-et)/5 - LTGt', 'LTGt - LTGt-2'),
    ('eps_forecast_error_3', 'eps_LTG_revision_3', 12, '(et+3-et)/3 - LTGt', 'LTGt - LTGt-3'),
    ('eps_forecast_error_4', 'eps_LTG_revision_3', 16, '(et+4-et)/4 - LTGt', 'LTGt - LTGt-3'),
    ('eps_forecast_error_5', 'eps_LTG_revision_3', 20, '(et+5-et)/5 - LTGt', 'LTGt - LTGt-3'),
]

eps_results = []

for fe_col, rev_col, lags, label, rev_label in horizons_eps:
    result = run_cg_regression(eps_ltg_all, fe_col, rev_col, lags, label, rev_label)
    if result:
        model, n_obs = result
        valid = eps_ltg_all[[fe_col, rev_col, "estimate_date"]].dropna()
#        print(f"Sample: {valid['estimate_date'].min().date()} to {valid['estimate_date'].max().date()}")
        eps_results.append({
            'revision_var': rev_label,
            'horizon': label,
            'coefficient': model.params[rev_col],
            'std_error': model.bse[rev_col],
            't_stat': model.tvalues[rev_col],
            'p_value': model.pvalues[rev_col],
            'r_squared': model.rsquared,
            'n_obs': n_obs
        })

        print(f"\n{rev_label} | {label}")
        print(f"Coefficient: {model.params[rev_col]:.4f}")
        print(f"Std Error (NW): ({model.bse[rev_col]:.4f})")
        print(f"t-statistic: {model.tvalues[rev_col]:.4f}")
        print(f"p-value: {model.pvalues[rev_col]:.4f}")
        print(f"R²: {model.rsquared:.2%}")
        print(f"Observations: {n_obs}")

print("\n" + "="*80)
print("Summary Table")
print("="*80)

for rev_var in ['LTGt - LTGt-1', 'LTGt - LTGt-2', 'LTGt - LTGt-3']:
    print(f"\n{rev_var}")
    print("-" * 80)

    eps_subset = [r for r in eps_results if r['revision_var'] == rev_var]

    if eps_subset:
        summary_df = pd.DataFrame({
            'Horizon': ['3-year', '4-year', '5-year'],
            'EPS Coef': [f"{r['coefficient']:.4f}" for r in eps_subset],
            'EPS SE': [f"({r['std_error']:.4f})" for r in eps_subset],
            'EPS t-stat': [f"{r['t_stat']:.2f}" for r in eps_subset],
            'EPS N': [r['n_obs'] for r in eps_subset],
        })

        print(summary_df.to_string(index=False))