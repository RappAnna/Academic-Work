import pandas as pd
import numpy as np

# Construct PD and PE ratio

# Get the constructed dividend and earnings index
price_ratios = (
    all_comp_agg[["quarter_end", "dividend_index", "earnings_index"]]
    .merge(
        sp500_quarterly_crsp[["quarter_end", "index_level"]],
        on="quarter_end",
        how="inner"
    )
)

# Calculate price ratios
price_ratios['pd_ratio'] = price_ratios['index_level'] / price_ratios['dividend_index']
price_ratios['pe_ratio'] = price_ratios['index_level'] / price_ratios['earnings_index']
price_ratios['de_ratio'] = price_ratios['dividend_index'] / price_ratios['earnings_index']

# Convert to logs 
price_ratios['log_pd'] = np.log(price_ratios['pd_ratio'])
price_ratios['log_pe'] = np.log(price_ratios['pe_ratio'])
price_ratios['log_de'] = np.log(price_ratios['de_ratio'])

# Prepare data
price_ratios = price_ratios.replace([np.inf, -np.inf], np.nan)
cols = price_ratios.columns.difference(["quarter_end"])
price_ratios = price_ratios.dropna(subset=cols)
price_ratios = price_ratios.loc[(price_ratios[cols] != 0).all(axis=1)].copy()


# Standard Deviation
subperiods = {
    '2003Q1-2023Q3': ('2003-01-01', '2023-09-30'),
    '2003Q1-2015Q3': ('2003-01-01', '2015-09-30'),
}

print("="*60)
print("STD DEV OF LOG PRICE RATIOS BY SUBPERIOD")
print("="*60)

for label, (start, end) in subperiods.items():
    mask = (price_ratios['quarter_end'] >= start) & (price_ratios['quarter_end'] <= end)
    sub = price_ratios.loc[mask]
    print(f"\n{label}  (N={len(sub)})")
    print(f"  Log P/D Std Dev: {sub['log_pd'].std():.4f}")
    print(f"  Log P/E Std Dev: {sub['log_pe'].std():.4f}")