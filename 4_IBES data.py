import wrds
import pandas as pd
from datetime import datetime
import numpy as np
import yfinance as yf

# Load Ibes Data (DPS, EPS)

db = wrds.Connection()

dps_start = '2003-01-01'
dps_end = '2023-12-31'
eps_start = '1976-01-01'
eps_end = '2023-12-31'

dps_data = db.raw_sql(f"""
    SELECT ticker, statpers as estimate_date, fpedats as fiscal_period_end,
           fpi, medest as forecast
    FROM ibes.statsumu_xepsus
    WHERE statpers >= '{dps_start}' AND statpers <= '{dps_end}'
        AND measure = 'DPS'
        AND fpi IN ('1', '2', '3', '4', '5', '6', '7', '8', '9')
""")

eps_data = db.raw_sql(f"""
    SELECT ticker, statpers as estimate_date, fpedats as fiscal_period_end,
           fpi, medest as forecast
    FROM ibes.statsumu_epsus 
    WHERE statpers >= '{eps_start}' AND statpers <= '{eps_end}'
        AND measure = 'EPS'
        AND fpi IN ('1', '2', '3', '4', '5', '6', '7', '8', '9')
""")

# Link Ibes ticker to permno 
ibes_link = db.raw_sql("""
    SELECT ticker, cusip, sdates
    FROM ibes.idsum
    WHERE ticker IS NOT NULL
        AND cusip IS NOT NULL
        AND sdates IS NOT NULL
""")

crsp_names = db.raw_sql("""
    SELECT permno, ncusip, namedt, nameenddt
    FROM crsp.stocknames
    WHERE ncusip IS NOT NULL
        AND permno IS NOT NULL
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
    lambda x: pd.date_range(x['link_start'], x['link_end'], freq='Q'),
    axis=1
)
dup_check = ibes_link.explode('date_range').groupby(['ticker', 'date_range'])['permno'].nunique()

ibes_link = ibes_link.drop(columns=['date_range']).sort_values(['ticker', 'link_start'], ascending=[True, False])

# Merge DPS data with permno
dps_data['estimate_date'] = pd.to_datetime(dps_data['estimate_date'])
dps_data['fiscal_period_end'] = pd.to_datetime(dps_data['fiscal_period_end'])

initial_dps_count = len(dps_data)
initial_dps_tickers = dps_data['ticker'].nunique()

dps_data = dps_data.merge(ibes_link, on='ticker', how='left')
dps_data = dps_data[
    (dps_data['estimate_date'] >= dps_data['link_start']) &
    (dps_data['estimate_date'] <= dps_data['link_end'])
].copy()

dps_data = dps_data.sort_values(
    ['ticker', 'estimate_date', 'fiscal_period_end', 'fpi', 'link_start'],
    ascending=[True, True, True, True, False]
).drop_duplicates(subset=['ticker', 'estimate_date', 'fiscal_period_end', 'fpi'], keep='first')

dps_data = dps_data.drop(columns=['link_start', 'link_end'])

# Merge EPS data with permno
eps_data['estimate_date'] = pd.to_datetime(eps_data['estimate_date'])
eps_data['fiscal_period_end'] = pd.to_datetime(eps_data['fiscal_period_end'])

initial_eps_count = len(eps_data)
initial_eps_tickers = eps_data['ticker'].nunique()

eps_data = eps_data.merge(ibes_link, on='ticker', how='left')
eps_data = eps_data[
    (eps_data['estimate_date'] >= eps_data['link_start']) &
    (eps_data['estimate_date'] <= eps_data['link_end'])
].copy()

eps_data = eps_data.sort_values(
    ['ticker', 'estimate_date', 'fiscal_period_end', 'fpi', 'link_start'],
    ascending=[True, True, True, True, False]
).drop_duplicates(subset=['ticker', 'estimate_date', 'fiscal_period_end', 'fpi'], keep='first')

eps_data = eps_data.drop(columns=['link_start', 'link_end'])

# Create quarterly time series for each permno
def create_quarterly_grid(data_df, start_date='2003-01-01', end_date='2023-12-31'):
    permno_dates = data_df.groupby(['permno', 'ticker'])['estimate_date'].agg(['min', 'max']).reset_index()
    
    overall_start = pd.to_datetime(start_date)
    overall_end = pd.to_datetime(end_date)
    
    permno_dates['min'] = permno_dates['min'].clip(lower=overall_start)
    permno_dates['max'] = permno_dates['max'].clip(upper=overall_end)
    
    quarterly_data = []
    for _, row in permno_dates.iterrows():
        quarters = pd.date_range(
            start=row['min'],
            end=row['max'],
            freq='Q'  
        )
        
        for quarter in quarters:
            quarter_end = quarter.normalize()
            quarterly_data.append({
                'permno': row['permno'],
                'ticker': row['ticker'],
                'quarter_end': quarter_end
            })
    
    return pd.DataFrame(quarterly_data)

dps_quarterly = create_quarterly_grid(dps_data, dps_start, dps_end)
eps_quarterly = create_quarterly_grid(eps_data, eps_start, eps_end)

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

def filter_sp500_quarterly(quarterly_df, sp500_df, min_days=30):
    quarterly_df = quarterly_df.copy()
    quarterly_df['in_sp500'] = False
    
    quarterly_df['quarter_start'] = quarterly_df['quarter_end'] - pd.offsets.QuarterBegin(1, startingMonth=1)
    
    for _, row in sp500_df.iterrows():
        permno = row['permno']
        for i in range(1, 5):
            start_date = row[f'start_{i}']
            end_date = row[f'ending_{i}']
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            
            mask = (
                (quarterly_df['permno'] == permno) &
                (start_date <= quarterly_df['quarter_end']) &
                (end_date >= quarterly_df['quarter_start'])
            )
            
            if min_days is not None and mask.any():
                overlap_start = quarterly_df.loc[mask, 'quarter_start'].clip(lower=start_date)
                overlap_end = quarterly_df.loc[mask, 'quarter_end'].clip(upper=end_date)
                overlap_days = (overlap_end - overlap_start).dt.days + 1
                
                mask_with_threshold = mask.copy()
                mask_with_threshold.loc[mask] = overlap_days >= min_days
                quarterly_df.loc[mask_with_threshold, 'in_sp500'] = True
            else:
                quarterly_df.loc[mask, 'in_sp500'] = True
    
    quarterly_df = quarterly_df.drop(columns=['quarter_start'])
    result = quarterly_df[quarterly_df['in_sp500']].drop(columns=['in_sp500'])
    
    return result

dps_quarterly_sp500 = filter_sp500_quarterly(dps_quarterly, sp500_constituents, MIN_DAYS_THRESHOLD)
eps_quarterly_sp500 = filter_sp500_quarterly(eps_quarterly, sp500_constituents, MIN_DAYS_THRESHOLD)


def find_latest_estimate_date_for_all_horizons(
    quarterly_grid_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    lookback_days: int = 90
) -> pd.DataFrame:
    import time
    start_time = time.time()
    
    qg = quarterly_grid_df.copy()
    qg["quarter_end"] = pd.to_datetime(qg["quarter_end"]).dt.normalize()
    
    fc = forecasts_df.copy()
    fc["estimate_date"] = pd.to_datetime(fc["estimate_date"]).dt.normalize()
    
    estimates_unique = fc[['permno', 'estimate_date']].drop_duplicates()
    
    all_results = []
    unique_permnos = qg['permno'].unique()
    
    
    for i, permno in enumerate(unique_permnos):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(unique_permnos)}")
        
        qg_permno = qg[qg['permno'] == permno].copy()
        est_permno = estimates_unique[estimates_unique['permno'] == permno].copy()
        
        if len(est_permno) == 0:
            continue
        
        qg_permno = qg_permno.sort_values('quarter_end').reset_index(drop=True)
        est_permno = est_permno.sort_values('estimate_date').reset_index(drop=True)
        
        merged_permno = pd.merge_asof(
            qg_permno[['ticker', 'quarter_end']],
            est_permno[['estimate_date']],
            left_on='quarter_end',
            right_on='estimate_date',
            direction='backward',
            tolerance=pd.Timedelta(days=lookback_days)
        )
        
        merged_permno['permno'] = permno
        merged_permno = merged_permno.rename(columns={'estimate_date': 'estimate_date_used'})
        
        all_results.append(merged_permno)
    
    result = pd.concat(all_results, ignore_index=True)
    result = result.dropna(subset=['estimate_date_used'])
    
    elapsed = time.time() - start_time
    
    return result[['permno', 'ticker', 'quarter_end', 'estimate_date_used']]


# Interpolate 
def interpolate_with_consistent_estimate_date(
    forecasts_df: pd.DataFrame,
    fixed_dates_df: pd.DataFrame,
    target_horizon_months: int = 12
) -> pd.DataFrame:
    import time
    start_time = time.time()
    
    df = forecasts_df.copy()
    df["estimate_date"] = pd.to_datetime(df["estimate_date"])
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    
    df = df[df["fiscal_period_end"] > df["estimate_date"]].copy()
    df = df[df["fpi"].isin(["1", "2", "3", "4", "5"])].copy()
    
    fixed = fixed_dates_df.copy()
    fixed["quarter_end"] = pd.to_datetime(fixed["quarter_end"]).dt.normalize()
    fixed["estimate_date_used"] = pd.to_datetime(fixed["estimate_date_used"]).dt.normalize()
        
    merged = fixed.merge(
        df[['permno', 'estimate_date', 'fpi', 'fiscal_period_end', 'forecast']],
        left_on=['permno', 'estimate_date_used'],
        right_on=['permno', 'estimate_date'],
        how='left'
    ).drop(columns=['estimate_date'])
    
    merged['days_ahead'] = (merged['fiscal_period_end'] - merged['quarter_end']).dt.days
    
    target_days = target_horizon_months * 30.4375
    
    lower = merged[merged['days_ahead'] <= target_days].copy()
    lower = lower.sort_values(['permno', 'quarter_end', 'days_ahead'])
    lower = lower.groupby(['permno', 'quarter_end'], as_index=False).last()
    lower = lower.rename(columns={
        'forecast': 'forecast_lower',
        'days_ahead': 'days_ahead_lower',
        'fpi': 'fpi_lower'
    })
    
    upper = merged[merged['days_ahead'] >= target_days].copy()
    upper = upper.sort_values(['permno', 'quarter_end', 'days_ahead'])
    upper = upper.groupby(['permno', 'quarter_end'], as_index=False).first()
    upper = upper.rename(columns={
        'forecast': 'forecast_upper',
        'days_ahead': 'days_ahead_upper',
        'fpi': 'fpi_upper'
    })
    
    result = lower.merge(
        upper[['permno', 'quarter_end', 'forecast_upper', 'days_ahead_upper', 'fpi_upper']],
        on=['permno', 'quarter_end'],
        how='inner'
    )
    
    same_point = result['days_ahead_lower'] == result['days_ahead_upper']
    result['forecast'] = np.nan
    result['method'] = ''
    
    if same_point.sum() > 0:
        result.loc[same_point, 'forecast'] = result.loc[same_point, 'forecast_lower']
        result.loc[same_point, 'method'] = 'annual_exact'
    
    not_same = ~same_point
    if not_same.sum() > 0:
        weight = (target_days - result.loc[not_same, 'days_ahead_lower']) / (
            result.loc[not_same, 'days_ahead_upper'] - result.loc[not_same, 'days_ahead_lower']
        )
        result.loc[not_same, 'forecast'] = (
            result.loc[not_same, 'forecast_lower'] + 
            weight * (result.loc[not_same, 'forecast_upper'] - result.loc[not_same, 'forecast_lower'])
        )
        result.loc[not_same, 'method'] = (
            'annual_' + result.loc[not_same, 'fpi_lower'].astype(str) + 
            '_' + result.loc[not_same, 'fpi_upper'].astype(str)
        )
    
    result['target_horizon_months'] = target_horizon_months
    result = result[['permno', 'ticker', 'quarter_end', 'target_horizon_months',
                     'forecast', 'method', 'estimate_date_used']]
    result = result[np.isfinite(result['forecast'])]
    
    elapsed = time.time() - start_time
    
    return result

eps_fixed_dates = find_latest_estimate_date_for_all_horizons(
    quarterly_grid_df=eps_quarterly_sp500,
    forecasts_df=eps_data,
    lookback_days=90
)

eps_1yr = interpolate_with_consistent_estimate_date(
    forecasts_df=eps_data,
    fixed_dates_df=eps_fixed_dates,
    target_horizon_months=12
)

eps_2yr = interpolate_with_consistent_estimate_date(
    forecasts_df=eps_data,
    fixed_dates_df=eps_fixed_dates,
    target_horizon_months=24
)

dps_fixed_dates = find_latest_estimate_date_for_all_horizons(
    quarterly_grid_df=dps_quarterly_sp500,
    forecasts_df=dps_data,
    lookback_days=90
)

dps_1yr = interpolate_with_consistent_estimate_date(
    forecasts_df=dps_data,
    fixed_dates_df=dps_fixed_dates,
    target_horizon_months=12
)

dps_2yr = interpolate_with_consistent_estimate_date(
    forecasts_df=dps_data,
    fixed_dates_df=dps_fixed_dates,
    target_horizon_months=24
)


# Transform the df
def transform_df(df: pd.DataFrame, years_ahead: int) -> pd.DataFrame:
    out = df.copy()
    
    out = out.drop(columns=["estimate_date_used"], errors="ignore")
    
    out = out.rename(columns={"quarter_end": "estimate_date"})
    out["estimate_date"] = pd.to_datetime(out["estimate_date"]).dt.normalize()

    out["quarter_end"] = out["estimate_date"] + pd.DateOffset(years=years_ahead)
    out["quarter_end"] = pd.to_datetime(out["quarter_end"]).dt.normalize()

    return out

eps_1yr = transform_df(eps_1yr.copy(), years_ahead=1)
dps_1yr = transform_df(dps_1yr.copy(), years_ahead=1)
eps_2yr = transform_df(eps_2yr.copy(), years_ahead=2)
dps_2yr = transform_df(dps_2yr.copy(), years_ahead=2)