import wrds
import pandas as pd
from datetime import datetime
import numpy as np
import yfinance as yf

# Aggregate IBES LTG indices

# Download IBES Forecast Data
db = wrds.Connection()

dps_start = '2003-01-01'
dps_end = '2023-09-30'
eps_start = '1976-01-01'
eps_end = '2023-09-30'

dps_ltg_data = db.raw_sql(f"""
    SELECT ticker, statpers as estimate_date, fpi, medest as forecast
    FROM ibes.statsumu_xepsus
    WHERE statpers >= '{dps_start}' AND statpers <= '{dps_end}'
      AND measure = 'DPS'
      AND fpi IN ('0')
""")

eps_ltg_data = db.raw_sql(f"""
    SELECT ticker, statpers as estimate_date, fpi, medest as forecast
    FROM ibes.statsumu_epsus
    WHERE statpers >= '{eps_start}' AND statpers <= '{eps_end}'
      AND measure = 'EPS'
      AND fpi IN ('0')
""")

# Linking IBES Ticker to Permno
ibes_link = db.raw_sql("""
    SELECT ticker, cusip, sdates
    FROM ibes.idsum
    WHERE ticker IS NOT NULL AND cusip IS NOT NULL AND sdates IS NOT NULL
""")

crsp_names = db.raw_sql("""
    SELECT permno, ncusip, namedt, nameenddt
    FROM crsp.stocknames
    WHERE ncusip IS NOT NULL AND permno IS NOT NULL
""")

db.close()

ibes_link['start_date'] = pd.to_datetime(ibes_link['sdates'], errors='coerce')
ibes_link['end_date'] = pd.Timestamp('2099-12-31')
ibes_link['cusip8'] = ibes_link['cusip'].astype(str).str[:8]

crsp_names['namedt'] = pd.to_datetime(crsp_names['namedt'], errors='coerce')
crsp_names['nameenddt'] = pd.to_datetime(crsp_names['nameenddt'], errors='coerce').fillna(pd.Timestamp('2099-12-31'))
crsp_names['cusip8'] = crsp_names['ncusip'].astype(str).str[:8]

ibes_link = ibes_link.merge(crsp_names, on='cusip8', how='inner')
ibes_link['link_start'] = ibes_link[['start_date', 'namedt']].max(axis=1)
ibes_link['link_end'] = ibes_link[['end_date', 'nameenddt']].min(axis=1)
ibes_link = ibes_link[ibes_link['link_start'] <= ibes_link['link_end']]
ibes_link = ibes_link[['ticker', 'permno', 'link_start', 'link_end']].drop_duplicates()

ibes_link['date_range'] = ibes_link.apply(
    lambda x: pd.date_range(x['link_start'], x['link_end'], freq='Q'), axis=1
)
dup_check = ibes_link.explode('date_range').groupby(['ticker', 'date_range'])['permno'].nunique()

# Merge earnings and dividend data with permno
dps_ltg_data['estimate_date'] = pd.to_datetime(dps_ltg_data['estimate_date'])

initial_dps_count = len(dps_ltg_data)
initial_dps_tickers = dps_ltg_data['ticker'].nunique()

dps_ltg_data = dps_ltg_data.merge(ibes_link, on='ticker', how='left')
dps_ltg_data = dps_ltg_data[
    (dps_ltg_data['estimate_date'] >= dps_ltg_data['link_start']) &
    (dps_ltg_data['estimate_date'] <= dps_ltg_data['link_end'])
].copy()

dps_ltg_data = dps_ltg_data.sort_values(
    ['ticker', 'estimate_date', 'fpi', 'link_start'],
    ascending=[True, True, True, False]
).drop_duplicates(subset=['ticker', 'estimate_date', 'fpi'], keep='first')

dps_ltg_data = dps_ltg_data.drop(columns=['link_start', 'link_end'])

eps_ltg_data['estimate_date'] = pd.to_datetime(eps_ltg_data['estimate_date'])

initial_eps_count = len(eps_ltg_data)
initial_eps_tickers = eps_ltg_data['ticker'].nunique()

eps_ltg_data = eps_ltg_data.merge(ibes_link, on='ticker', how='left')
eps_ltg_data = eps_ltg_data[
    (eps_ltg_data['estimate_date'] >= eps_ltg_data['link_start']) &
    (eps_ltg_data['estimate_date'] <= eps_ltg_data['link_end'])
].copy()

eps_ltg_data = eps_ltg_data.sort_values(
    ['ticker', 'estimate_date', 'fpi', 'link_start'],
    ascending=[True, True, True, False]
).drop_duplicates(subset=['ticker', 'estimate_date', 'fpi'], keep='first')

eps_ltg_data = eps_ltg_data.drop(columns=['link_start', 'link_end'])

# Filter for S&P 500 constituents
sp500_path = r"D:\Code\constituents.csv"
sp500_constituents = pd.read_csv(
    sp500_path,
    usecols=['permno', 'start_1', 'ending_1', 'start_2', 'ending_2',
             'start_3', 'ending_3', 'start_4', 'ending_4']
)

date_columns = ['start_1', 'ending_1', 'start_2', 'ending_2',
                'start_3', 'ending_3', 'start_4', 'ending_4']
for col in date_columns:
    sp500_constituents[col] = pd.to_datetime(sp500_constituents[col], errors='coerce')

MIN_DAYS_THRESHOLD = 30

dps_ltg_data['_quarter_end']   = dps_ltg_data['estimate_date'].dt.to_period('Q').dt.end_time.dt.normalize()
dps_ltg_data['_quarter_start'] = dps_ltg_data['_quarter_end'] - pd.offsets.QuarterBegin(1, startingMonth=1)
dps_ltg_data['in_sp500'] = False

for _, row in sp500_constituents.iterrows():
    permno = row['permno']
    for i in range(1, 5):
        start_date = row[f'start_{i}']
        end_date   = row[f'ending_{i}']
        if pd.isna(start_date) or pd.isna(end_date):
            continue

        mask = (
            (dps_ltg_data['permno'] == permno) &
            (start_date <= dps_ltg_data['_quarter_end']) &
            (end_date   >= dps_ltg_data['_quarter_start'])
        )

        if MIN_DAYS_THRESHOLD is not None and mask.any():
            overlap_start = dps_ltg_data.loc[mask, '_quarter_start'].clip(lower=start_date)
            overlap_end   = dps_ltg_data.loc[mask, '_quarter_end'].clip(upper=end_date)
            overlap_days  = (overlap_end - overlap_start).dt.days + 1
            mask_with_threshold = mask.copy()
            mask_with_threshold.loc[mask] = overlap_days >= MIN_DAYS_THRESHOLD
            dps_ltg_data.loc[mask_with_threshold, 'in_sp500'] = True
        else:
            dps_ltg_data.loc[mask, 'in_sp500'] = True

dps_ltg_data = dps_ltg_data.drop(columns=['_quarter_start', '_quarter_end'])
dps_ltg_data = dps_ltg_data[dps_ltg_data['in_sp500']].drop(columns=['in_sp500'])

eps_ltg_data['_quarter_end']   = eps_ltg_data['estimate_date'].dt.to_period('Q').dt.end_time.dt.normalize()
eps_ltg_data['_quarter_start'] = eps_ltg_data['_quarter_end'] - pd.offsets.QuarterBegin(1, startingMonth=1)
eps_ltg_data['in_sp500'] = False

for _, row in sp500_constituents.iterrows():
    permno = row['permno']
    for i in range(1, 5):
        start_date = row[f'start_{i}']
        end_date   = row[f'ending_{i}']
        if pd.isna(start_date) or pd.isna(end_date):
            continue

        mask = (
            (eps_ltg_data['permno'] == permno) &
            (start_date <= eps_ltg_data['_quarter_end']) &
            (end_date   >= eps_ltg_data['_quarter_start'])
        )

        if MIN_DAYS_THRESHOLD is not None and mask.any():
            overlap_start = eps_ltg_data.loc[mask, '_quarter_start'].clip(lower=start_date)
            overlap_end   = eps_ltg_data.loc[mask, '_quarter_end'].clip(upper=end_date)
            overlap_days  = (overlap_end - overlap_start).dt.days + 1
            mask_with_threshold = mask.copy()
            mask_with_threshold.loc[mask] = overlap_days >= MIN_DAYS_THRESHOLD
            eps_ltg_data.loc[mask_with_threshold, 'in_sp500'] = True
        else:
            eps_ltg_data.loc[mask, 'in_sp500'] = True

eps_ltg_data = eps_ltg_data.drop(columns=['_quarter_start', '_quarter_end'])
eps_ltg_data = eps_ltg_data[eps_ltg_data['in_sp500']].drop(columns=['in_sp500'])

# Convert to quarterly data
dps_ltg_data = (
    dps_ltg_data
    .sort_values(["permno", "estimate_date"])
    .groupby(["permno", pd.Grouper(key="estimate_date", freq="Q")], as_index=False)
    .tail(1)
)

eps_ltg_data = (
    eps_ltg_data
    .sort_values(["permno", "estimate_date"])
    .groupby(["permno", pd.Grouper(key="estimate_date", freq="Q")], as_index=False)
    .tail(1)
)

dps_ltg_data["estimate_date"] = dps_ltg_data["estimate_date"].dt.to_period("Q").dt.end_time.dt.normalize()
eps_ltg_data["estimate_date"] = eps_ltg_data["estimate_date"].dt.to_period("Q").dt.end_time.dt.normalize()

# Merge with Compustat Dataframe
compustat = compustat.copy()
compustat["quarter_end"] = pd.to_datetime(compustat["quarter_end"])

dps_ltg_data = dps_ltg_data.copy()
eps_ltg_data = eps_ltg_data.copy()

comp_cols = ["permno", "quarter_end", "mktcap"]
comp_sub  = compustat[comp_cols].drop_duplicates(subset=["permno", "quarter_end"])

dps_ltg_data = dps_ltg_data.merge(
    comp_sub,
    left_on=["permno", "estimate_date"],
    right_on=["permno", "quarter_end"],
    how="left"
).drop(columns=["quarter_end"])

eps_ltg_data = eps_ltg_data.merge(
    comp_sub,
    left_on=["permno", "estimate_date"],
    right_on=["permno", "quarter_end"],
    how="left"
).drop(columns=["quarter_end"])

# Create firm-level and index Forecast
eps_ltg_data["estimate_date"] = pd.to_datetime(eps_ltg_data["estimate_date"])
dps_ltg_data["estimate_date"] = pd.to_datetime(dps_ltg_data["estimate_date"])

eps_ltg_agg = (
    eps_ltg_data
    .groupby(pd.Grouper(key="estimate_date", freq="Q"))["mktcap"]
    .sum()
    .reset_index()
)

dps_ltg_agg = (
    dps_ltg_data
    .groupby(pd.Grouper(key="estimate_date", freq="Q"))["mktcap"]
    .sum()
    .reset_index()
)

# Calculate weighted firm-level forecasts and aggregate
dps_ltg_agg = dps_ltg_agg.rename(columns={"mktcap": "mktcap_total"})

dps_ltg_data = dps_ltg_data.merge(
    dps_ltg_agg[["estimate_date", "mktcap_total"]],
    on="estimate_date",
    how="left"
)

dps_ltg_data["forecast_firm"] = (
    (dps_ltg_data["forecast"] * dps_ltg_data["mktcap"]) / dps_ltg_data["mktcap_total"]
)

dps_ltg_agg = (
    dps_ltg_data
    .groupby("estimate_date")["forecast_firm"]
    .sum()
    .reset_index()
)

eps_ltg_agg = eps_ltg_agg.rename(columns={"mktcap": "mktcap_total"})

eps_ltg_data = eps_ltg_data.merge(
    eps_ltg_agg[["estimate_date", "mktcap_total"]],
    on="estimate_date",
    how="left"
)

eps_ltg_data["forecast_firm"] = (
    (eps_ltg_data["forecast"] * eps_ltg_data["mktcap"]) / eps_ltg_data["mktcap_total"]
)

eps_ltg_agg = (
    eps_ltg_data
    .groupby("estimate_date")["forecast_firm"]
    .sum()
    .reset_index()
)

eps_ltg_agg = eps_ltg_agg.rename(columns={'forecast_firm': 'forecast_ltg'})
dps_ltg_agg = dps_ltg_agg.rename(columns={'forecast_firm': 'forecast_ltg'})

# Prepare dataset for regressions
eps_ltg_agg["estimate_date_t+5"] = eps_ltg_agg["estimate_date"] + pd.DateOffset(years=5)
eps_ltg_agg["estimate_date_t-1"] = eps_ltg_agg["estimate_date"] - pd.DateOffset(years=1)
eps_ltg_agg["estimate_date_t-2"] = eps_ltg_agg["estimate_date"] - pd.DateOffset(years=2)
eps_ltg_agg["estimate_date_t-3"] = eps_ltg_agg["estimate_date"] - pd.DateOffset(years=3)

dps_ltg_agg["estimate_date_t+5"] = dps_ltg_agg["estimate_date"] + pd.DateOffset(years=5)
dps_ltg_agg["estimate_date_t-1"] = dps_ltg_agg["estimate_date"] - pd.DateOffset(years=1)
dps_ltg_agg["estimate_date_t-2"] = dps_ltg_agg["estimate_date"] - pd.DateOffset(years=2)
dps_ltg_agg["estimate_date_t-3"] = dps_ltg_agg["estimate_date"] - pd.DateOffset(years=3)

eps_ltg_agg['forecast_ltg'] = eps_ltg_agg['forecast_ltg'] / 100
dps_ltg_agg['forecast_ltg'] = dps_ltg_agg['forecast_ltg'] / 100