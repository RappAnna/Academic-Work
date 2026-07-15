import wrds
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. WRDS CONNECTION AND DATA EXTRACTION 
# ==============================================================================
print("Connecting to WRDS...")
db = wrds.Connection(wrds_username='your_username')

# Define date range for quarterly data
start_date = '2010-01-01'
end_date = '2015-01-01'

# Query to get all quarterly EPS estimates
print("\nExtracting EPS data from IBES...")
eps_data = db.raw_sql(f"""
    SELECT ticker, estimator, cusip, anndats, actdats, fpi, fpedats, 
           value as estimated_value, actual,
           anntims, acttims
    FROM ibes.det_epsus
    WHERE anndats >= '{start_date}' 
      AND anndats <= '{end_date}'
      AND fpi IN ('6', '7', '8', '9')
      AND measure = 'EPS'
      AND value IS NOT NULL
      AND usfirm = 1
      AND curr_act = 'USD'
    ORDER BY ticker, fpedats, anndats
""")

print(f"  ✓ {len(eps_data):,} EPS records extracted")
print(f"  ✓ {eps_data['ticker'].nunique()} unique tickers")


# ==============================================================================
# 2. LINK IBES TICKER TO PERMNO VIA CUSIP
# ==============================================================================
print("\nLinking IBES tickers to PERMNOs via CUSIP...")

# Get IBES identifier mapping
ibes_ids = db.raw_sql("""
    SELECT ticker, cusip, sdates
    FROM ibes.idsum
    WHERE ticker IS NOT NULL
        AND cusip IS NOT NULL
        AND sdates IS NOT NULL
""")

# Get CRSP stock names with PERMNO-CUSIP links
crsp_names = db.raw_sql("""
    SELECT permno, ncusip, namedt, nameenddt
    FROM crsp.stocknames
    WHERE ncusip IS NOT NULL
        AND permno IS NOT NULL
""")

db.close()
print("WRDS connection closed")

# Process dates
ibes_ids['start_date'] = pd.to_datetime(ibes_ids['sdates'], errors='coerce')
ibes_ids['end_date'] = pd.Timestamp('2099-12-31')

crsp_names['namedt'] = pd.to_datetime(crsp_names['namedt'], errors='coerce')
crsp_names['nameenddt'] = pd.to_datetime(crsp_names['nameenddt'], errors='coerce').fillna(pd.Timestamp('2099-12-31'))

# Standardize CUSIP to 8 characters
ibes_ids['cusip8'] = ibes_ids['cusip'].astype(str).str[:8]
crsp_names['cusip8'] = crsp_names['ncusip'].astype(str).str[:8]

# Merge and filter overlapping periods
ibes_link = ibes_ids.merge(crsp_names, on='cusip8', how='inner')
ibes_link['link_start'] = ibes_link[['start_date', 'namedt']].max(axis=1)
ibes_link['link_end'] = ibes_link[['end_date', 'nameenddt']].min(axis=1)
ibes_link = ibes_link[ibes_link['link_start'] <= ibes_link['link_end']]
ibes_link = ibes_link[['ticker', 'permno', 'link_start', 'link_end']].drop_duplicates()

# Check for ambiguous links
ibes_link_test = ibes_link.copy()
ibes_link_test['date_range'] = ibes_link_test.apply(
    lambda x: pd.date_range(x['link_start'], x['link_end'], freq='Q'),
    axis=1
)
ibes_link_exploded = ibes_link_test.explode('date_range')
dup_tickers = ibes_link_exploded.groupby(['ticker', 'date_range'])['permno'].nunique()

if (dup_tickers > 1).any():
    print(f" ⚠ WARNING: {(dup_tickers > 1).sum()} ticker-date pairs link to multiple PERMNOs")
    print("   → Keeping most recent link_start for each ticker-date")

# For each ticker-date, keep the most recent link
ibes_link = ibes_link.sort_values(['ticker', 'link_start'], ascending=[True, False])

print(f" ✓ {len(ibes_link):,} I/B/E/S-CRSP links created")
print(f" ✓ {ibes_link['ticker'].nunique()} unique tickers")
print(f" ✓ {ibes_link['permno'].nunique()} unique PERMNOs")

# ==============================================================================
# 3. MERGE EPS DATA WITH PERMNO
# ==============================================================================
print("\nMerging EPS data with PERMNOs...")

# Rename columns for consistency
eps_data = eps_data.rename(columns={
    'anndats': 'estimate_date',
    'fpedats': 'fiscal_period_end',
    'actdats': 'actual_date'
})

# Convert dates to datetime
eps_data['estimate_date'] = pd.to_datetime(eps_data['estimate_date'])
eps_data['fiscal_period_end'] = pd.to_datetime(eps_data['fiscal_period_end'])
eps_data['actual_date'] = pd.to_datetime(eps_data['actual_date'], errors='coerce')

# Merge with ibes_link
eps_data = eps_data.merge(
    ibes_link[['ticker', 'permno', 'link_start', 'link_end']], 
    on='ticker', 
    how='left'
)

# Filter to only keep links valid at the estimate_date
eps_data = eps_data[
    (eps_data['estimate_date'] >= eps_data['link_start']) &
    (eps_data['estimate_date'] <= eps_data['link_end'])
].copy()

# Handle duplicates: keep the most recent link (highest link_start)
eps_data = eps_data.sort_values(
    ['ticker', 'estimate_date', 'fiscal_period_end', 'fpi', 'estimator', 'link_start'],
    ascending=[True, True, True, True, True, False]
)
eps_data = eps_data.drop_duplicates(
    subset=['ticker', 'estimate_date', 'fiscal_period_end', 'fpi', 'estimator'], 
    keep='first'
)

# Drop the linking columns we no longer need
eps_data = eps_data.drop(columns=['link_start', 'link_end'])

# ==============================================================================
# 4. ADD S&P 500 MEMBERSHIP FLAG
# ==============================================================================
print("\nAdding S&P 500 membership flag...")

# Load S&P 500 constituents file
sp500_path = r"C:\Users\Anna\Documents\GitHub\IBES_data\2_Data\2_1_Input\constituents_31122023_edit.csv"
sp500_constituents = pd.read_csv(
    sp500_path,
    usecols=['permno', 'start_1', 'ending_1', 'start_2', 'ending_2', 
             'start_3', 'ending_3', 'start_4', 'ending_4']
)

# Convert date columns to datetime
date_columns = ['start_1', 'ending_1', 'start_2', 'ending_2', 
                'start_3', 'ending_3', 'start_4', 'ending_4']
for col in date_columns:
    sp500_constituents[col] = pd.to_datetime(sp500_constituents[col], errors='coerce')

# Initialize the S&P 500 flag
eps_data['in_sp500'] = False

print(f"  Processing {len(sp500_constituents)} S&P 500 constituents...")

# Flag observations where the firm was in S&P 500 at the estimate date
for idx, row in sp500_constituents.iterrows():
    permno = row['permno']
    
    for i in range(1, 5):
        start_date = row[f'start_{i}']
        end_date = row[f'ending_{i}']
        
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        mask = (
            (eps_data['permno'] == permno) &
            (eps_data['estimate_date'] >= start_date) &
            (eps_data['estimate_date'] <= end_date)
        )
        eps_data.loc[mask, 'in_sp500'] = True

# Summary statistics
n_sp500_records = eps_data['in_sp500'].sum()
n_total_records = len(eps_data)
pct_sp500 = (n_sp500_records / n_total_records * 100) if n_total_records > 0 else 0


# ==============================================================================
# 5. KEEP ONLY LATEST FORECAST PER ANALYST-FIRM-PERIOD-HORIZON (WITH TIMING)
# ==============================================================================
print("\nKeeping only the latest forecast per analyst-firm-period-horizon...")
print("  Applying forecast horizon timing rules:")
print("    FPI=6: Latest forecast in same quarter as fiscal_period_end")
print("    FPI=7: Latest forecast 1 quarter before fiscal_period_end")
print("    FPI=8: Latest forecast 2 quarters before fiscal_period_end")
print("    FPI=9: Latest forecast 3 quarters before fiscal_period_end")

# Store count before filtering
records_before = len(eps_data)
print(f"\n  Before filtering: {records_before:,} records")

# Create quarter identifiers for estimate_date and fiscal_period_end
eps_data['estimate_quarter'] = eps_data['estimate_date'].dt.to_period('Q')
eps_data['fiscal_quarter'] = eps_data['fiscal_period_end'].dt.to_period('Q')

# Calculate the expected quarter based on FPI
# FPI=6 means same quarter (0 quarters back)
# FPI=7 means 1 quarter back, etc.
eps_data['quarters_back'] = eps_data['fpi'].astype(int) - 6
eps_data['expected_quarter'] = eps_data['fiscal_quarter'] - eps_data['quarters_back']

# Filter to keep only forecasts made in the correct quarter for their FPI
eps_data_filtered = eps_data[
    eps_data['estimate_quarter'] == eps_data['expected_quarter']
].copy()

print(f"  After timing filter: {len(eps_data_filtered):,} records")
print(f"  Removed: {records_before - len(eps_data_filtered):,} records with incorrect timing")

# Now keep only the latest forecast within each valid quarter
# Sort by estimate_date in descending order to get the latest first
eps_data_filtered = eps_data_filtered.sort_values(
    ['permno', 'fiscal_period_end', 'fpi', 'estimator', 'estimate_date'],
    ascending=[True, True, True, True, False]
)

# Keep only the latest forecast (first after sorting) for each combination
eps_data_filtered = eps_data_filtered.drop_duplicates(
    subset=['permno', 'fiscal_period_end', 'fpi', 'estimator'],
    keep='first'
)

# Drop the helper columns
eps_data_filtered = eps_data_filtered.drop(
    columns=['estimate_quarter', 'fiscal_quarter', 'quarters_back', 'expected_quarter']
)

# Replace original dataframe
eps_data = eps_data_filtered

# Store count after all filtering
records_after = len(eps_data)
print(f"  After duplicate removal: {records_after:,} records")
print(f"  Total removed: {records_before - records_after:,} records")

# Update summary statistics after filtering
n_sp500_records = eps_data['in_sp500'].sum()
n_total_records = len(eps_data)
pct_sp500 = (n_sp500_records / n_total_records * 100) if n_total_records > 0 else 0

print(f"\n  ✓ {n_sp500_records:,} records ({pct_sp500:.1f}%) flagged as S&P 500")
print(f"  ✓ {eps_data[eps_data['in_sp500']]['permno'].nunique()} unique S&P 500 PERMNOs")

# Show distribution by FPI
print(f"\n  Distribution by forecast horizon:")
for fpi in sorted(eps_data['fpi'].unique()):
    count = (eps_data['fpi'] == fpi).sum()
    print(f"    FPI {fpi}: {count:,} records")


# ==============================================================================
# 6. CREATE FORECAST ERROR
# ==============================================================================

# Create forecast revision (actual - forecast)
eps_data['forecast_error'] = eps_data['actual'] - eps_data['estimated_value']

eps_copy = eps_data

# ==============================================================================
# 7. CREATE FORECAST REVISION (CONSECUTIVE FPI ONLY)
# ==============================================================================
print("\nCreating forecast revision column (consecutive FPI only)...")

# Sort the data to ensure proper ordering
eps_data = eps_data.sort_values(
    ['permno', 'estimator', 'fiscal_period_end', 'fpi'],
    ascending=[True, True, True, False]  # FPI descending so we go from 9→8→7→6
)

# Create lagged FPI and lagged estimated_value
eps_data['prev_fpi'] = eps_data.groupby(
    ['permno', 'estimator', 'fiscal_period_end']
)['fpi'].shift(1)

eps_data['prev_estimated_value'] = eps_data.groupby(
    ['permno', 'estimator', 'fiscal_period_end']
)['estimated_value'].shift(1)

# Calculate expected previous FPI (should be exactly FPI + 1)
eps_data['expected_prev_fpi'] = eps_data['fpi'].astype(int) + 1

# Create forecast revision ONLY if the previous FPI is exactly FPI + 1
# This ensures we only calculate: FPI=6 - FPI=7, FPI=7 - FPI=8, FPI=8 - FPI=9
eps_data['forecast_revision'] = np.where(
    eps_data['prev_fpi'].astype(str) == eps_data['expected_prev_fpi'].astype(str),
    eps_data['estimated_value'] - eps_data['prev_estimated_value'],
    np.nan
)

# Count how many revisions we have
n_revisions = eps_data['forecast_revision'].notna().sum()
n_total = len(eps_data)
print(f"  ✓ Created {n_revisions:,} forecast revisions out of {n_total:,} total records")
print(f"  ✓ {n_total - n_revisions:,} records have no consecutive previous forecast")

# Show summary statistics
print(f"\n  Forecast Revision Summary Statistics:")
print(f"    Mean: {eps_data['forecast_revision'].mean():.4f}")
print(f"    Median: {eps_data['forecast_revision'].median():.4f}")
print(f"    Std Dev: {eps_data['forecast_revision'].std():.4f}")
print(f"    Min: {eps_data['forecast_revision'].min():.4f}")
print(f"    Max: {eps_data['forecast_revision'].max():.4f}")

# Show distribution by FPI
print(f"\n  Forecast Revisions by FPI:")
for fpi in sorted(eps_data['fpi'].unique()):
    fpi_revisions = eps_data[eps_data['fpi'] == fpi]['forecast_revision'].notna().sum()
    fpi_total = (eps_data['fpi'] == fpi).sum()
    print(f"    FPI {fpi}: {fpi_revisions:,} revisions out of {fpi_total:,} records")

# Drop helper columns
eps_data = eps_data.drop(columns=['prev_fpi', 'prev_estimated_value', 'expected_prev_fpi'])

print("\n  Forecast revision calculation complete!")
print("  ✓ Only consecutive FPI pairs used (6-7, 7-8, 8-9)")
print("  ✓ Non-consecutive pairs (e.g., 6-8) are excluded")