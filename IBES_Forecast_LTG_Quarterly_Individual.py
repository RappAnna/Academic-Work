import wrds
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. WRDS CONNECTION AND DATA EXTRACTION 
# ==============================================================================
print("Connecting to WRDS...")
db = wrds.Connection(wrds_username='your_username')

# Define date range for LTG data
start_date = '2010-01-01'
end_date = '2015-01-01'


# LTG query
ltg_data = db.raw_sql(f"""
    SELECT ticker, estimator, cusip, anndats, fpi, fpedats, 
           value as estimated_value
    FROM ibes.det_epsus
    WHERE anndats >= '{start_date}' 
      AND anndats <= '{end_date}'
      AND measure = 'EPS'
      AND fpi = '0'
      AND value IS NOT NULL
    ORDER BY ticker, estimator, anndats
""")


print(f"  ✓ {len(ltg_data):,} LTG records extracted")
print(f"  ✓ {ltg_data['ticker'].nunique()} unique tickers")

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
# 3. MERGE LTG DATA WITH PERMNO
# ==============================================================================
print("\nMerging LTG data with PERMNOs...")

# Rename columns for consistency
ltg_data = ltg_data.rename(columns={
    'anndats': 'estimate_date',
})

# Convert dates to datetime
ltg_data['estimate_date'] = pd.to_datetime(ltg_data['estimate_date'])

# Merge with ibes_link
ltg_data = ltg_data.merge(
    ibes_link[['ticker', 'permno', 'link_start', 'link_end']], 
    on='ticker', 
    how='left'
)

# Filter to only keep links valid at the estimate_date
ltg_data = ltg_data[
    (ltg_data['estimate_date'] >= ltg_data['link_start']) &
    (ltg_data['estimate_date'] <= ltg_data['link_end'])
].copy()

# Handle duplicates: keep the most recent link (highest link_start)
ltg_data = ltg_data.sort_values(
    ['ticker', 'estimate_date', 'estimator', 'link_start'],
    ascending=[True, True, True, False]
)
ltg_data = ltg_data.drop_duplicates(
    subset=['ticker', 'estimate_date', 'estimator'], 
    keep='first'
)

# Drop the linking columns we no longer need
ltg_data = ltg_data.drop(columns=['link_start', 'link_end'])

print(f"  ✓ {len(ltg_data):,} LTG records with PERMNO")
print(f"  ✓ {ltg_data['permno'].nunique()} unique PERMNOs")

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
ltg_data['in_sp500'] = False

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
            (ltg_data['permno'] == permno) &
            (ltg_data['estimate_date'] >= start_date) &
            (ltg_data['estimate_date'] <= end_date)
        )
        ltg_data.loc[mask, 'in_sp500'] = True

# Summary statistics
n_sp500_records = ltg_data['in_sp500'].sum()
n_total_records = len(ltg_data)
pct_sp500 = (n_sp500_records / n_total_records * 100) if n_total_records > 0 else 0

print(f"\n  ✓ {n_sp500_records:,} records ({pct_sp500:.1f}%) flagged as S&P 500")
print(f"  ✓ {ltg_data[ltg_data['in_sp500']]['permno'].nunique()} unique S&P 500 PERMNOs")

# ==============================================================================
# 5. KEEP ONLY LATEST FORECAST PER ANALYST-FIRM-QUARTER
# ==============================================================================
print("\nKeeping only the latest LTG forecast per analyst-firm-quarter...")

# Store count before filtering
records_before = len(ltg_data)
print(f"\n  Before filtering: {records_before:,} records")

# Create quarter identifier for estimate_date
ltg_data['estimate_quarter'] = ltg_data['estimate_date'].dt.to_period('Q')

# Sort by estimate_date in descending order to get the latest first
ltg_data = ltg_data.sort_values(
    ['permno', 'estimator', 'estimate_quarter', 'estimate_date'],
    ascending=[True, True, True, False]
)

# Keep only the latest forecast (first after sorting) for each combination
ltg_data = ltg_data.drop_duplicates(
    subset=['permno', 'estimator', 'estimate_quarter'],
    keep='first'
)

# Store count after filtering
records_after = len(ltg_data)
print(f"  After filtering: {records_after:,} records")
print(f"  Total removed: {records_before - records_after:,} records")

print(f"\n  ✓ Latest LTG forecast per analyst-firm-quarter retained")

# ==============================================================================
# 6. CREATE ANNUAL LTG FORECAST REVISION (4 QUARTERS BACK)
# ==============================================================================
print("\nCreating annual LTG forecast revision...")

# Sort the data to ensure proper ordering
ltg_data = ltg_data.sort_values(
    ['permno', 'estimator', 'estimate_quarter'],
    ascending=[True, True, True]
)

# Create lagged quarter (4 quarters back = 1 year)
ltg_data['prev_quarter'] = ltg_data.groupby(
    ['permno', 'estimator']
)['estimate_quarter'].shift(4)

# Create lagged estimated_value (4 quarters back)
ltg_data['prev_estimated_value'] = ltg_data.groupby(
    ['permno', 'estimator']
)['estimated_value'].shift(4)

# Calculate expected previous quarter (should be exactly 4 quarters earlier)
ltg_data['expected_prev_quarter'] = ltg_data['estimate_quarter'] - 4

# Create annual forecast revision ONLY if the previous quarter is exactly 4 quarters back
# This ensures we only calculate year-over-year changes in the same quarter
ltg_data['forecast_revision'] = np.where(
    ltg_data['prev_quarter'] == ltg_data['expected_prev_quarter'],
    ltg_data['estimated_value'] - ltg_data['prev_estimated_value'],
    np.nan
)

# Count how many revisions we have
n_revisions = ltg_data['forecast_revision'].notna().sum()
n_total = len(ltg_data)
print(f"  ✓ Created {n_revisions:,} annual LTG revisions out of {n_total:,} total records")
print(f"  ✓ {n_total - n_revisions:,} records have no forecast exactly 4 quarters earlier")

# Show summary statistics
print(f"\n  Annual LTG Revision Summary Statistics:")
revision_stats = ltg_data['forecast_revision'].dropna()
if len(revision_stats) > 0:
    print(f"    Mean: {revision_stats.mean():.4f}")
    print(f"    Median: {revision_stats.median():.4f}")
    print(f"    Std Dev: {revision_stats.std():.4f}")
    print(f"    Min: {revision_stats.min():.4f}")
    print(f"    Max: {revision_stats.max():.4f}")
else:
    print("    No revisions available")

# Drop helper columns
ltg_data = ltg_data.drop(columns=['prev_quarter', 'prev_estimated_value', 'expected_prev_quarter'])

print("\n  Annual LTG forecast revision calculation complete!")
print("  ✓ Only year-over-year changes (4 quarters apart) calculated")
print("  ✓ Same quarter comparisons (e.g., Q1 2013 - Q1 2012)")


    

