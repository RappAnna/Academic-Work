# Create the realized indices for IBES covered data

# Filter dataframes to only rows with forecasts
eps_1yr = eps_1yr[eps_1yr['forecast'].notna()].copy()
dps_1yr = dps_1yr[dps_1yr['forecast'].notna()].copy()
eps_1yr_filt = eps_1yr.sort_values(["permno", "quarter_end", "estimate_date"]).drop_duplicates(["permno", "quarter_end"], keep="last")
dps_1yr_filt = dps_1yr.sort_values(["permno", "quarter_end", "estimate_date"]).drop_duplicates(["permno", "quarter_end"], keep="last")

compustat = compustat.sort_values(["permno", "quarter_end", "mktcap"]).drop_duplicates(["permno", "quarter_end"], keep="last")

# Firm-level earnings and dividends
ibes_sp500_eps = compustat.copy()
ibes_sp500_dps = compustat.copy()

def add_4q_sum_strict(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    df = df.sort_values(["permno", "quarter_end"]).copy()

    q = pd.PeriodIndex(pd.to_datetime(df["quarter_end"]).dt.to_period("Q"), freq="Q")
    df["_qnum"] = q.year * 4 + q.quarter

    g = df.groupby("permno", group_keys=False)

    df[out_col] = g[value_col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)

    df["_qnum_lag3"] = g["_qnum"].shift(3)
    is_consecutive_4q = (df["_qnum"] - df["_qnum_lag3"] == 3)
    df.loc[~is_consecutive_4q, out_col] = np.nan

    required = [out_col]
    if out_col == "earnings_4q" and "dividends_4q" in df.columns:
        required.append("dividends_4q")
    if out_col == "dividends_4q" and "earnings_4q" in df.columns:
        required.append("earnings_4q")

    df = df.dropna(subset=required).copy()
    
    df = df[df["quarter_end"] >= pd.Timestamp("1976-01-01")].copy()

    return df.drop(columns=["_qnum", "_qnum_lag3"])

ibes_sp500_eps = add_4q_sum_strict(ibes_sp500_eps, value_col="earnings",  out_col="earnings_4q")
ibes_sp500_dps = add_4q_sum_strict(ibes_sp500_dps, value_col="dividends", out_col="dividends_4q")

ibes_sp500_eps = ibes_sp500_eps.merge(
    eps_1yr_filt[["permno","quarter_end","forecast"]], 
    on=["permno","quarter_end"], 
    how="left", 
    validate="m:1"
)

ibes_sp500_dps = ibes_sp500_dps.merge(
    dps_1yr_filt[["permno","quarter_end","forecast"]], 
    on=["permno","quarter_end"], 
    how="left", 
    validate="m:1"
)

ibes_sp500_eps = ibes_sp500_eps[ibes_sp500_eps['forecast'].notna()].copy()
ibes_sp500_dps = ibes_sp500_dps[ibes_sp500_dps['forecast'].notna()].copy()

# Create aggregate dataframes with scaling 
# Aggregate to S&P 500 level
ibes_sp500_eps = ibes_sp500_eps.groupby('quarter_end').agg({
    'earnings_4q': 'sum',
    'mktcap': 'sum',
    'permno': 'count'
}).reset_index()

ibes_sp500_dps = ibes_sp500_dps.groupby('quarter_end').agg({
    'dividends_4q': 'sum',
    'mktcap': 'sum',
    'permno': 'count'
}).reset_index()

ibes_sp500_eps.rename(columns={
    'earnings_4q': 'total_earnings_4q_ibes',
    'mktcap': 'total_mktcap_ibes',
    'permno': 'num_firms_ibes'
}, inplace=True)

ibes_sp500_dps.rename(columns={
    'dividends_4q': 'total_dividends_4q_ibes',
    'mktcap': 'total_mktcap_ibes',
    'permno': 'num_firms_ibes'
}, inplace=True)


# Create Aggregate S&P 500 index
ibes_sp500_eps = ibes_sp500_eps.merge(
    sp500_quarterly_crsp[['quarter_end', 'divisor']],
    on='quarter_end',
    how='inner'
)

ibes_sp500_dps = ibes_sp500_dps.merge(
    sp500_quarterly_crsp[['quarter_end', 'divisor']],
    on='quarter_end',
    how='inner'
)

ibes_sp500_eps['total_earnings_4q_ibes'] = ibes_sp500_eps['total_earnings_4q_ibes'] / ibes_sp500_eps['divisor']
ibes_sp500_dps['total_dividends_4q_ibes'] = ibes_sp500_dps['total_dividends_4q_ibes'] / ibes_sp500_dps['divisor']

ibes_sp500_eps = ibes_sp500_eps.merge(
    all_comp_agg[['quarter_end', 'total_mktcap']],
    on='quarter_end',
    how='left'
)

ibes_sp500_dps = ibes_sp500_dps.merge(
    all_comp_agg[['quarter_end', 'total_mktcap']],
    on='quarter_end',
    how='left'
)

# Calculate the scaling ratio and adjust earnings and dividends series
ibes_sp500_eps['scaling_ratio'] = ibes_sp500_eps['total_mktcap'] / ibes_sp500_eps['total_mktcap_ibes']
ibes_sp500_eps['earnings_index'] = ibes_sp500_eps['total_earnings_4q_ibes'] * ibes_sp500_eps['scaling_ratio']

ibes_sp500_dps['scaling_ratio'] = ibes_sp500_dps['total_mktcap'] / ibes_sp500_dps['total_mktcap_ibes']
ibes_sp500_dps['dividends_index'] = ibes_sp500_dps['total_dividends_4q_ibes'] * ibes_sp500_dps['scaling_ratio']


# Calculate growth rates
ibes_sp500_eps = ibes_sp500_eps.sort_values("quarter_end").reset_index(drop=True)
ibes_sp500_dps = ibes_sp500_dps.sort_values("quarter_end").reset_index(drop=True)

def log_growth_4q(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")  
    s_lead = s.shift(-4)
    valid = (s > 0) & (s_lead > 0)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[valid] = np.log(s_lead.loc[valid]) - np.log(s.loc[valid])
    return out

ibes_sp500_dps["div_index_growth"] = log_growth_4q(ibes_sp500_dps["dividends_index"])
ibes_sp500_eps["ear_index_growth"] = log_growth_4q(ibes_sp500_eps["earnings_index"])