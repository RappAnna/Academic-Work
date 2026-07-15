all_comp_sp500 = compustat

# Create firm-level dividends and earnings

all_comp_sp500["shares_out_scaled"] = all_comp_sp500["shares_outstanding"] * 1_000_000

all_comp_sp500["earnings"]  = all_comp_sp500["shares_out_scaled"] * all_comp_sp500["eps"]
all_comp_sp500["dividends"] = all_comp_sp500["shares_out_scaled"] * all_comp_sp500["div_per_share"]
all_comp_sp500["mktcap"]    = all_comp_sp500["shares_out_scaled"] * all_comp_sp500["price"]

all_comp_sp500 = all_comp_sp500.sort_values(["permno", "quarter_end"]).reset_index(drop=True)

def calculate_trailing_4q_properly(df):
    """
    Calculate trailing 4Q sums, but only output rows where we have valid data
    """
    df = df.sort_values(["permno", "quarter_end"]).copy()
    
    q = pd.PeriodIndex(pd.to_datetime(df["quarter_end"]), freq="Q")
    df["_qnum"] = q.year * 4 + q.quarter
    
    g = df.groupby("permno", group_keys=False)
    
    df["earnings_4q"] = g["earnings"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    df["dividends_4q"] = g["dividends"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    
    df["_qnum_lag3"] = g["_qnum"].shift(3)
    is_consecutive_4q = (df["_qnum"] - df["_qnum_lag3"] == 3)
    
    df["earnings_4q"] = np.where(is_consecutive_4q, df["earnings_4q"], np.nan)
    df["dividends_4q"] = np.where(is_consecutive_4q, df["dividends_4q"], np.nan)
    
    df = df[df["earnings_4q"].notna() & df["dividends_4q"].notna()].copy()
    
    df = df[df["quarter_end"] >= pd.Timestamp("1976-01-01")].copy()
    
    df = df.drop(columns=["_qnum", "_qnum_lag3"])
    return df

all_comp_sp500 = calculate_trailing_4q_properly(all_comp_sp500)

# Create aggregate S&P 500 index
all_comp_agg = all_comp_sp500.groupby('quarter_end').agg({
    'dividends_4q': 'sum',
    'earnings_4q': 'sum',
    'mktcap': 'sum',
    'permno': 'count'
}).reset_index()

all_comp_agg.rename(columns={
    'dividends_4q': 'total_dividends_4q',
    'earnings_4q': 'total_earnings_4q',
    'mktcap': 'total_mktcap',
    'permno': 'num_firms'
}, inplace=True)

# Calculate dividend and earnings indices
all_comp_agg = all_comp_agg.merge(
    sp500_quarterly_crsp[['quarter_end', 'divisor']],
    on='quarter_end',
    how='inner'
)

all_comp_agg['dividend_index'] = all_comp_agg['total_dividends_4q'] / all_comp_agg['divisor']
all_comp_agg['earnings_index'] = all_comp_agg['total_earnings_4q'] / all_comp_agg['divisor']

# Calculate growth rates
all_comp_agg = all_comp_agg.sort_values('quarter_end').reset_index(drop=True)

def ann_log_growth_k_years(series, k_years: int):
    s = pd.to_numeric(series, errors="coerce")
    s_fwd = s.shift(-4 * k_years)
    valid = (s > 0) & (s_fwd > 0)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[valid] = (np.log(s_fwd.loc[valid]) - np.log(s.loc[valid])) / k_years
    return out


all_comp_agg["div_gr_yoy"] = ann_log_growth_k_years(all_comp_agg["dividend_index"], 1)
all_comp_agg["ear_gr_yoy"] = ann_log_growth_k_years(all_comp_agg["earnings_index"], 1)

for k in [3, 4, 5]:
    all_comp_agg[f"div_gr_{k}y_ann"] = ann_log_growth_k_years(all_comp_agg["dividend_index"], k)
    all_comp_agg[f"ear_gr_{k}y_ann"] = ann_log_growth_k_years(all_comp_agg["earnings_index"], k)