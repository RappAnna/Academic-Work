# Correlation Check

# Load Shiller (2015) data
shiller_path = r"D:\Code\Shiller_data.xls"
shiller_data = pd.read_excel(
    shiller_path,
    sheet_name='Data',
    skiprows=7
)

date_raw = pd.to_numeric(shiller_data.iloc[:, 0], errors="coerce")
shiller_data = shiller_data.loc[date_raw.notna()].copy()
date_raw = date_raw.loc[date_raw.notna()]

shiller_data["year"] = np.floor(date_raw).astype(int)
shiller_data["month"] = np.round((date_raw - shiller_data["year"]) * 100).astype(int)

shiller_data["month"] = shiller_data["month"].clip(1, 12)

shiller_data["Date"] = pd.to_datetime(
    shiller_data["year"].astype(str) + "-" +
    shiller_data["month"].astype(str).str.zfill(2) + "-01",
    errors="coerce"
)

shiller_data = shiller_data.dropna(subset=["Date"])

shiller_data.rename(columns={
    shiller_data.columns[1]: 'P',
    shiller_data.columns[2]: 'D',  
    shiller_data.columns[3]: 'E'   
}, inplace=True)

shiller_data['quarter_end'] = shiller_data['Date'] + pd.offsets.QuarterEnd(0)

shiller_data_filtered = shiller_data[
    (shiller_data['Date'] >= START_DATE) & 
    (shiller_data['Date'] <= END_DATE)
].copy()

# Convert to quarterly by taking the last month of each quarter
shiller_quarterly = (
    shiller_data_filtered
    .sort_values('Date')
    .groupby('quarter_end', as_index=False)
    .last()  
    [['quarter_end', 'P', 'D', 'E']]
)

shiller_quarterly = shiller_quarterly.sort_values('quarter_end').reset_index(drop=True)

shiller_quarterly['quarter_end'] = pd.to_datetime(shiller_quarterly['quarter_end']).dt.normalize()

# Calculate year-over-year growth
shiller_quarterly['div_gr_yoy'] = (
    np.log(shiller_quarterly['D'].shift(-4)) -
    np.log(shiller_quarterly['D'])
)

shiller_quarterly['ear_gr_yoy'] = (
    np.log(shiller_quarterly['E'].shift(-4)) -
    np.log(shiller_quarterly['E'])
)


# Load SPY dividend data from CRSP with ordinary dividend filter
db = wrds.Connection()

spy_crsp = db.raw_sql(f'''
    SELECT 
        d.exdt as date,
        d.divamt as dividend,
        d.distcd
    FROM crsp.dsedist d
    WHERE d.permno = 84398
        AND d.exdt >= '{START_DATE}'
        AND d.exdt <= '{END_DATE}'
        AND d.divamt > 0
        AND d.distcd IN (1232, 1212)
''')

spy_crsp['date'] = pd.to_datetime(spy_crsp['date'])

spy_div_df = spy_crsp[['date', 'dividend']].copy()

if hasattr(spy_div_df["date"].dtype, 'tz') and spy_div_df["date"].dt.tz is not None:
    spy_div_df["date"] = spy_div_df["date"].dt.tz_localize(None)

spy_div_df["quarter_end"] = spy_div_df['date'] + pd.offsets.QuarterEnd(0)

# Aggregate to quarterly series
spy_quarterly = (
    spy_div_df
    .groupby("quarter_end", as_index=False)["dividend"]
    .sum()
    .rename(columns={"dividend": "spy_dividend_qtr"})
    .sort_values("quarter_end")
    .reset_index(drop=True)
)

# Calculate trailing 4-quarter sum
spy_quarterly = spy_quarterly.sort_values("quarter_end").reset_index(drop=True)

q = pd.PeriodIndex(spy_quarterly["quarter_end"], freq="Q")
spy_quarterly["_qnum"] = q.year * 4 + q.quarter

spy_quarterly["spy_dividend_4q"] = (
    spy_quarterly["spy_dividend_qtr"]
    .rolling(4, min_periods=4)
    .sum()
)

spy_quarterly["_qnum_lag3"] = spy_quarterly["_qnum"].shift(3)
is_consecutive_4q = (spy_quarterly["_qnum"] - spy_quarterly["_qnum_lag3"] == 3)

spy_quarterly.loc[~is_consecutive_4q, "spy_dividend_4q"] = np.nan

spy_quarterly = spy_quarterly.drop(columns=["_qnum", "_qnum_lag3"])

# Calculate YoY growth
spy_quarterly['spy_div_growth_yoy'] = (
    np.log(spy_quarterly['spy_dividend_4q'].shift(-4)) -
    np.log(spy_quarterly['spy_dividend_4q'])
)

spy_quarterly['quarter_end'] = pd.to_datetime(spy_quarterly['quarter_end']).dt.normalize()

db.close()

# Normalization of all dates 
all_comp_agg["quarter_end"] = pd.to_datetime(all_comp_agg["quarter_end"]).dt.normalize()
ibes_sp500_dps["quarter_end"] = pd.to_datetime(ibes_sp500_dps["quarter_end"]).dt.normalize()
ibes_sp500_eps["quarter_end"] = pd.to_datetime(ibes_sp500_eps["quarter_end"]).dt.normalize()
shiller_quarterly["quarter_end"] = pd.to_datetime(shiller_quarterly["quarter_end"]).dt.normalize()
spy_quarterly["quarter_end"] = pd.to_datetime(spy_quarterly["quarter_end"]).dt.normalize()


# Table 1: Correlations
print("\n" + "="*80)
print("Correlations of S&P 500 Dividend Measures")
print("="*80)

print("\n" + "-"*80)
print("Dividend Levels - Full Sample")
print("-"*80)

# Merge all dividend series
div_levels_full = (
    all_comp_agg[["quarter_end", "total_dividends_4q"]]
    .merge(
        ibes_sp500_dps[["quarter_end", "total_dividends_4q_ibes"]],
        on="quarter_end",
        how="inner"
    )
    .merge(
        shiller_quarterly[["quarter_end", "D"]],
        on="quarter_end",
        how="inner"
    )
    .merge(
        spy_quarterly[["quarter_end", "spy_dividend_4q"]],
        on="quarter_end",
        how="inner"
    )
)

div_levels_full = div_levels_full.rename(columns={
    "total_dividends_4q": "All_Companies",
    "total_dividends_4q_ibes": "IBES",
    "D": "Shiller",
    "spy_dividend_4q": "SPY"
})

div_levels_full = div_levels_full.sort_values('quarter_end').reset_index(drop=True)

print(f"\nFull sample period: {div_levels_full['quarter_end'].min().date()} to {div_levels_full['quarter_end'].max().date()}")
print(f"N = {len(div_levels_full)} quarters\n")

corr_div_levels_full = div_levels_full[["All_Companies", "IBES", "Shiller", "SPY"]].corr()

print("Correlation Matrix (Levels - Full Sample):")
print(corr_div_levels_full.round(3))

print("\n" + "-"*80)
print("Dividend Levels - PAPER SUBSAMPLE (2003Q1-2015Q3)")
print("-"*80)

div_levels_paper = div_levels_full[
    (div_levels_full['quarter_end'] >= '2003-01-01') & 
    (div_levels_full['quarter_end'] <= '2015-09-30')
].copy()

print(f"\nPaper sample period: {div_levels_paper['quarter_end'].min().date()} to {div_levels_paper['quarter_end'].max().date()}")
print(f"N = {len(div_levels_paper)} quarters\n")

corr_div_levels_paper = div_levels_paper[["All_Companies", "IBES", "Shiller", "SPY"]].corr()

print("Correlation Matrix (Levels - Paper Sample):")
print(corr_div_levels_paper.round(3))


print("\n" + "-"*80)
print("Dividend Growth (Year-over-Year) - FULL SAMPLE")
print("-"*80)

# Merge all dividend growth series
div_growth_full = (
    all_comp_agg[["quarter_end", "div_gr_yoy"]]
    .merge(
        ibes_sp500_dps[["quarter_end", "div_index_growth"]],
        on="quarter_end",
        how="inner"
    )
    .merge(
        shiller_quarterly[["quarter_end", "div_gr_yoy"]],
        on="quarter_end",
        how="inner",
        suffixes=("_all", "_shiller")
    )
    .merge(
        spy_quarterly[["quarter_end", "spy_div_growth_yoy"]],
        on="quarter_end",
        how="inner"
    )
)

div_growth_full = div_growth_full.rename(columns={
    "div_gr_yoy_all": "All_Companies",
    "div_index_growth": "IBES",
    "div_gr_yoy_shiller": "Shiller",
    "spy_div_growth_yoy": "SPY"
})

div_growth_full = div_growth_full.sort_values('quarter_end').reset_index(drop=True)

div_growth_full = div_growth_full.dropna()

print(f"\nFull sample period: {div_growth_full['quarter_end'].min().date()} to {div_growth_full['quarter_end'].max().date()}")
print(f"N = {len(div_growth_full)} quarters (after removing NaN)\n")

corr_div_growth_full = div_growth_full[["All_Companies", "IBES", "Shiller", "SPY"]].corr()

print("Correlation Matrix (Growth - Full Sample):")
print(corr_div_growth_full.round(3))

print("\n" + "-"*80)
print("Dividend Growth (Year-over-Year) - PAPER SUBSAMPLE (2003Q1-2015Q3)")
print("-"*80)

# Filter to paper's sample period
div_growth_paper = div_growth_full[
    (div_growth_full['quarter_end'] >= '2003-01-01') & 
    (div_growth_full['quarter_end'] <= '2015-09-30')
].copy()

print(f"\nPaper sample period: {div_growth_paper['quarter_end'].min().date()} to {div_growth_paper['quarter_end'].max().date()}")
print(f"N = {len(div_growth_paper)} quarters (after removing NaN)\n")

# Calculate correlation matrix
corr_div_growth_paper = div_growth_paper[["All_Companies", "IBES", "Shiller", "SPY"]].corr()

print("Correlation Matrix (Growth - Paper Sample):")
print(corr_div_growth_paper.round(3))

print("\n" + "-"*80)
print("EARNINGS LEVELS - FULL SAMPLE")
print("-"*80)

# Merge all earnings series
earn_levels_full = (
    all_comp_agg[["quarter_end", "total_earnings_4q"]]
    .merge(
        ibes_sp500_eps[["quarter_end", "total_earnings_4q_ibes"]],
        on="quarter_end",
        how="inner"
    )
    .merge(
        shiller_quarterly[["quarter_end", "E"]],
        on="quarter_end",
        how="inner"
    )
)

earn_levels_full = earn_levels_full.rename(columns={
    "total_earnings_4q": "All_Companies",
    "total_earnings_4q_ibes": "IBES",
    "E": "Shiller"
})

earn_levels_full = earn_levels_full.sort_values('quarter_end').reset_index(drop=True)

print(f"\nFull sample period: {earn_levels_full['quarter_end'].min().date()} to {earn_levels_full['quarter_end'].max().date()}")
print(f"N = {len(earn_levels_full)} quarters\n")

corr_earn_levels_full = earn_levels_full[["All_Companies", "IBES", "Shiller"]].corr()

print("Correlation Matrix (Levels - Full Sample):")
print(corr_earn_levels_full.round(3))

print("\n" + "-"*80)
print("Earnings Levels - PAPER SUBSAMPLE (1976Q1-2015Q3)")
print("-"*80)

earn_levels_paper = earn_levels_full[
    (earn_levels_full['quarter_end'] >= '1976-01-01') & 
    (earn_levels_full['quarter_end'] <= '2015-09-30')
].copy()

print(f"\nPaper sample period: {earn_levels_paper['quarter_end'].min().date()} to {earn_levels_paper['quarter_end'].max().date()}")
print(f"N = {len(earn_levels_paper)} quarters\n")

corr_earn_levels_paper = earn_levels_paper[["All_Companies", "IBES", "Shiller"]].corr()

print("Correlation Matrix (Levels - Paper Sample):")
print(corr_earn_levels_paper.round(3))

print("\n" + "-"*80)
print("Earnings Growth (Year-over-Year) - FULL SAMPLE")
print("-"*80)

# Merge all earnings growth series
earn_growth_full = (
    all_comp_agg[["quarter_end", "ear_gr_yoy"]]
    .merge(
        ibes_sp500_eps[["quarter_end", "ear_index_growth"]],
        on="quarter_end",
        how="inner"
    )
    .merge(
        shiller_quarterly[["quarter_end", "ear_gr_yoy"]],
        on="quarter_end",
        how="inner",
        suffixes=("_all", "_shiller")
    )
)

earn_growth_full = earn_growth_full.rename(columns={
    "ear_gr_yoy_all": "All_Companies",
    "ear_index_growth": "IBES",
    "ear_gr_yoy_shiller": "Shiller"
})

earn_growth_full = earn_growth_full.sort_values('quarter_end').reset_index(drop=True)

earn_growth_full = earn_growth_full.dropna()

print(f"\nFull sample period: {earn_growth_full['quarter_end'].min().date()} to {earn_growth_full['quarter_end'].max().date()}")
print(f"N = {len(earn_growth_full)} quarters (after removing NaN)\n")

corr_earn_growth_full = earn_growth_full[["All_Companies", "IBES", "Shiller"]].corr()

print("Correlation Matrix (Growth - Full Sample):")
print(corr_earn_growth_full.round(3))

print("\n" + "-"*80)
print("Earnings Growth (Year-over-Year) - PAPER SUBSAMPLE (1976Q1-2015Q3)")
print("-"*80)

earn_growth_paper = earn_growth_full[
    (earn_growth_full['quarter_end'] >= '1976-01-01') & 
    (earn_growth_full['quarter_end'] <= '2015-09-30')
].copy()

print(f"\nPaper sample period: {earn_growth_paper['quarter_end'].min().date()} to {earn_growth_paper['quarter_end'].max().date()}")
print(f"N = {len(earn_growth_paper)} quarters (after removing NaN)\n")

corr_earn_growth_paper = earn_growth_paper[["All_Companies", "IBES", "Shiller"]].corr()

print("Correlation Matrix (Growth - Paper Sample):")
print(corr_earn_growth_paper.round(3))