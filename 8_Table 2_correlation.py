import wrds
import pandas as pd
from datetime import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Correlations check

# Load De La O and Myers (2021) data and prepare
delao_earnings_path = r"D:\Code\Earnings_growth_expectations.xlsx"

for header_row in [0, 1, 2]:
    print(f"\n  Trying header row {header_row}...")
    test_df = pd.read_excel(delao_earnings_path, header=header_row, nrows=5)
    print(f"    Columns: {list(test_df.columns)}")
    print(f"    First row: {list(test_df.iloc[0].values) if len(test_df) > 0 else 'Empty'}")
    
delao_earnings = pd.read_excel(delao_earnings_path, header=1)

delao_dividends_path = r"D:\Code\Dividend_growth_expectations.xlsx"

delao_dividends = pd.read_excel(delao_dividends_path, header=0)

delao_earnings.columns = [str(col).strip() if col is not None else f'col_{i}' 
                          for i, col in enumerate(delao_earnings.columns)]
delao_dividends.columns = [str(col).strip() if col is not None else f'col_{i}' 
                           for i, col in enumerate(delao_dividends.columns)]

delao_earnings["quarter_year"] = (
    pd.PeriodIndex(delao_earnings["Year"].astype(int).astype(str) + "Q" + delao_earnings["Quarter"].astype(int).astype(str),
                   freq="Q")
      .to_timestamp("Q")
      .normalize()
)

delao_dividends["quarter_year"] = (
    pd.PeriodIndex(delao_dividends["Year"].astype(int).astype(str) + "Q" + delao_dividends["Quarter"].astype(int).astype(str),
                   freq="Q")
      .to_timestamp("Q")
      .normalize()
)

cols_keep = [
    "quarter_year",
    "Expected one-year log earnings growth",
    "Realized next year log earnings growth",
    "Current price ratio",
]

delao_earnings = delao_earnings.loc[:, cols_keep].copy()
delao_dividends = delao_dividends.drop(columns=['Year', 'Quarter'], errors='ignore')

# Correlation check
# Filter to 2022Q1 
cutoff_date = pd.Timestamp('2022-03-31')

eps_1_index_comp = eps_1_index[eps_1_index['estimate_date'] <= cutoff_date].copy()
dps_1_index_comp = dps_1_index[dps_1_index['estimate_date'] <= cutoff_date].copy()

delao_earnings_comp  = delao_earnings[delao_earnings['quarter_year'] <= cutoff_date].copy()
delao_dividends_comp = delao_dividends[delao_dividends['quarter_year'] <= cutoff_date].copy()

print(f"\nFiltered to ≤ 2022Q1:")
print(f"  EPS index:       {len(eps_1_index_comp)} rows, last date: {eps_1_index_comp['estimate_date'].max()}")
print(f"  DPS index:       {len(dps_1_index_comp)} rows, last date: {dps_1_index_comp['estimate_date'].max()}")
print(f"  De La O earnings: {len(delao_earnings_comp)} rows, last date: {delao_earnings_comp['quarter_year'].max()}")
print(f"  De La O dividends:{len(delao_dividends_comp)} rows, last date: {delao_dividends_comp['quarter_year'].max()}")

# Earnings correlation
eps_comparison = eps_1_index_comp.merge(
    delao_earnings_comp[['quarter_year', 'Expected one-year log earnings growth']],
    left_on='estimate_date',
    right_on='quarter_year',
    how='inner'
)

corr_eps = eps_comparison['ear_growth_1yr'].corr(
    eps_comparison['Expected one-year log earnings growth']
)

print(f"\nEarnings Growth Correlation: {corr_eps:.4f}")
print(f"N observations: {eps_comparison['ear_growth_1yr'].notna().sum()}")
print(f"Date range: {eps_comparison['estimate_date'].min()} to {eps_comparison['estimate_date'].max()}")

# Dividend correlation 
dps_comparison = dps_1_index_comp.merge(
    delao_dividends_comp[['quarter_year', 'Expected one-year log dividend growth']],
    left_on='estimate_date',
    right_on='quarter_year',
    how='inner'
)

corr_dps = dps_comparison['div_growth_1yr'].corr(
    dps_comparison['Expected one-year log dividend growth']
)

print(f"\nDividend Growth Correlation: {corr_dps:.4f}")
print(f"N observations: {dps_comparison['div_growth_1yr'].notna().sum()}")
print(f"Date range: {dps_comparison['estimate_date'].min()} to {dps_comparison['estimate_date'].max()}")