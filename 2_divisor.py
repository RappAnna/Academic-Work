import numpy as np
import pandas as pd
import wrds
import yfinance as yf

START_DATE = "1975-01-01"
END_DATE = "2023-09-30"

# Calculate S&P 500 divisor

db = wrds.Connection()

# Pull CRSP monthly data (MSF) and compute market cap
crsp_msf = db.raw_sql(f"""
    SELECT
        permno,
        date,
        abs(prc) as prc,
        shrout
    FROM crsp.msf
    WHERE date >= '{START_DATE}'
      AND date <= '{END_DATE}'
      AND prc IS NOT NULL
      AND shrout IS NOT NULL
""")
crsp_msf['date'] = pd.to_datetime(crsp_msf['date'])
crsp_msf['mktcap'] = crsp_msf['prc'] * crsp_msf['shrout'] * 1000.0

# Construct quarter-end observation
crsp_msf['qtr'] = crsp_msf['date'].dt.to_period('Q')
crsp_msf = (
    crsp_msf.sort_values(['permno', 'date'])
            .groupby(['permno', 'qtr'], as_index=False)
            .tail(1)
)
crsp_msf['quarter_end'] = crsp_msf['qtr'].dt.to_timestamp('Q')
crsp_msf = crsp_msf.drop(columns=['qtr', 'date'])

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

crsp_msf['quarter_start'] = crsp_msf['quarter_end'] - pd.offsets.QuarterBegin(1, startingMonth=1)

crsp_msf['in_sp500'] = False

for _, row in sp500_constituents.iterrows():
    permno = row['permno']
    for i in range(1, 5):
        start_date = row[f'start_{i}']
        end_date = row[f'ending_{i}']
        
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        mask = (
            (crsp_msf['permno'] == permno) &
            (start_date <= crsp_msf['quarter_end']) &
            (end_date >= crsp_msf['quarter_start'])
        )
        
        if MIN_DAYS_THRESHOLD is not None and mask.any():
            overlap_start = crsp_msf.loc[mask, 'quarter_start'].clip(lower=start_date)
            overlap_end = crsp_msf.loc[mask, 'quarter_end'].clip(upper=end_date)
            overlap_days = (overlap_end - overlap_start).dt.days + 1
            
            mask_with_threshold = mask.copy()
            mask_with_threshold.loc[mask] = overlap_days >= MIN_DAYS_THRESHOLD
            crsp_msf.loc[mask_with_threshold, 'in_sp500'] = True
        else:
            crsp_msf.loc[mask, 'in_sp500'] = True

crsp_msf = crsp_msf.drop(columns=['quarter_start'])
crsp_msf = crsp_msf[crsp_msf['in_sp500']].drop(columns=['in_sp500'])

crsp_msf = crsp_msf.drop_duplicates(subset=['permno', 'quarter_end'])

# Aggregate total market cap at each calendar quarter end
sp500_quarterly_crsp = (
    crsp_msf.groupby('quarter_end', as_index=False)
            .agg(total_mktcap=('mktcap', 'sum'),
                 num_firms=('permno', 'count'))
            .sort_values('quarter_end')
)

# Pull S&P 500 index level at calendar quarter ends
sp500_index = yf.download("^GSPC", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
close = sp500_index["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

sp500_index_qtr = close.resample("Q").last().to_frame("index_level").reset_index()
sp500_index_qtr.rename(columns={'Date': 'quarter_end'}, inplace=True)
sp500_index_qtr['quarter_end'] = pd.to_datetime(sp500_index_qtr['quarter_end']).dt.normalize()

# Compute divisor
sp500_quarterly_crsp = sp500_quarterly_crsp.merge(sp500_index_qtr, on='quarter_end', how='inner')
sp500_quarterly_crsp['divisor'] = sp500_quarterly_crsp['total_mktcap'] / sp500_quarterly_crsp['index_level']

db.close()
