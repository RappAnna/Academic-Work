import pandas as pd
import numpy as np
import wrds
import yfinance as yf

# Downloading realized data

START_DATE = "1975-01-01"
END_DATE = "2023-09-30"

db = wrds.Connection()

# Load Compustat realized data
compustat = db.raw_sql(f"""
    SELECT 
        gvkey,
        datadate,
        datacqtr as quarter,
        prccq as price,
        epspxq as eps,
        cshoq as shares_outstanding,
        fqtr as fiscal_quarter,
        fyearq as fiscal_year
    FROM comp.fundq
    WHERE datadate >= '{START_DATE}'
        AND datadate <= '{END_DATE}'
        AND prccq IS NOT NULL
        AND cshoq IS NOT NULL
    ORDER BY gvkey, datadate
""")

compustat["datadate"] = pd.to_datetime(compustat["datadate"])
compustat["month_id"] = compustat["datadate"].dt.year * 100 + compustat["datadate"].dt.month
compustat["quarter_end"] = compustat["datadate"] + pd.offsets.QuarterEnd(0)
compustat["quarter_end"] = compustat["datadate"].dt.to_period("Q").dt.end_time.dt.normalize()

# Load CCM Linking Table
ccm_link = db.raw_sql("""
    SELECT 
        gvkey,
        lpermno as permno,
        lpermco as permco,
        linktype,
        linkprim,
        linkdt,
        linkenddt,
        usedflag
    FROM crsp.ccmxpf_linktable
    WHERE substr(linktype,1,1) = 'L'
        AND linkprim IN ('P','C')
        AND usedflag = 1
""")

ccm_link["linkdt"] = pd.to_datetime(ccm_link["linkdt"])
ccm_link["linkenddt"] = pd.to_datetime(ccm_link["linkenddt"]).fillna(pd.to_datetime("2099-12-31"))

# Merge Compustat with CCM Link
compustat = pd.merge(compustat, ccm_link, on="gvkey", how="left")
compustat = compustat[
    (compustat["quarter_end"] >= compustat["linkdt"]) &
    (compustat["quarter_end"] <= compustat["linkenddt"])
].copy()


duplicates = compustat.groupby(["gvkey", "datadate"]).size()
n_duplicates = (duplicates > 1).sum()
print(f"  {n_duplicates:,} (gvkey, datadate) pairs with multiple permno values")

linktype_rank = {"LC": 3, "LU": 2}  
compustat["linktype_score"] = compustat["linktype"].map(linktype_rank).fillna(1).astype(int)
compustat["linkprim_score"] = (compustat["linkprim"] == "P").astype(int)

compustat = (
    compustat
    .sort_values(
        ["gvkey", "datadate", "linkprim_score", "linktype_score", "linkdt", "permno"],
        ascending=[True, True, False, False, False, True]
    )
    .groupby(["gvkey", "datadate"], as_index=False)
    .first()
)

# Add Cusip
crsp_cusip = db.raw_sql(f"""
    SELECT 
        permno,
        ncusip as cusip,
        namedt,
        nameenddt
    FROM crsp.stocknames
    WHERE namedt <= '{END_DATE}'
        AND nameenddt >= '{START_DATE}'
        AND ncusip IS NOT NULL
""")

crsp_cusip["namedt"] = pd.to_datetime(crsp_cusip["namedt"])
crsp_cusip["nameenddt"] = pd.to_datetime(crsp_cusip["nameenddt"]).fillna(pd.to_datetime("2099-12-31"))

compustat = pd.merge(compustat, crsp_cusip, on="permno", how="left")

compustat = compustat[
    (compustat["quarter_end"] >= compustat["namedt"]) &
    (compustat["quarter_end"] <= compustat["nameenddt"])
].drop_duplicates(subset=["gvkey", "datadate"], keep="first")


# Download CRSP realized dividends
crsp_divs = db.raw_sql(f"""
    SELECT 
        d.permno,
        d.divamt,
        d.exdt,
        d.distcd,
        d.dclrdt,
        d.paydt,
        d.rcrddt,
        n.shrcd,
        n.exchcd
    FROM crsp.dsedist d
    LEFT JOIN crsp.dsenames n 
        ON d.permno = n.permno 
        AND d.exdt >= n.namedt 
        AND d.exdt <= n.nameendt
    WHERE d.exdt >= '{START_DATE}'
        AND d.exdt <= '{END_DATE}'
        AND d.divamt > 0
        AND d.distcd IS NOT NULL
""")

crsp_divs["exdt"] = pd.to_datetime(crsp_divs["exdt"])

print(f"  ✓ Loaded {len(crsp_divs):,} dividend records from CRSP")

# Filter out special dividends using distcd
ordinary_distcds = [1212, 1222, 1232, 1242, 1312, 1332, 1342, 1348]
before_filter = len(crsp_divs)
crsp_divs = crsp_divs[crsp_divs['distcd'].isin(ordinary_distcds)].copy()
print(f"  ✓ After distcd filter: {len(crsp_divs):,} ordinary dividend records")
print(f"  ✓ Excluded {before_filter - len(crsp_divs):,} special/non-ordinary dividends")

crsp_divs['quarter_end'] = crsp_divs['exdt'].dt.to_period('Q').dt.end_time.dt.normalize()

ordinary_divs = crsp_divs.groupby(['permno', 'quarter_end'])['divamt'].sum().reset_index()
ordinary_divs.rename(columns={'divamt': 'div_per_share'}, inplace=True)

compustat = pd.merge(
    compustat,
    ordinary_divs,
    on=['permno', 'quarter_end'],
    how='left'
)

compustat['div_per_share'] = compustat['div_per_share'].fillna(0)

compustat = compustat[[
    "gvkey", "permno", "permco", "cusip", "datadate", "month_id",
    "quarter", "fiscal_quarter", "fiscal_year", "quarter_end",
    "price", "div_per_share", "eps", "shares_outstanding",
    "linktype", "linkprim"
]].copy()

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

compustat['quarter_start'] = compustat['quarter_end'] - pd.offsets.QuarterBegin(1, startingMonth=1)

compustat['in_sp500'] = False

MIN_DAYS_THRESHOLD = 30  

for _, row in sp500_constituents.iterrows():
    permno = row['permno']
    for i in range(1, 5):
        start_date = row[f'start_{i}']
        end_date = row[f'ending_{i}']
        
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        mask = (
            (compustat['permno'] == permno) &
            (start_date <= compustat['quarter_end']) &
            (end_date >= compustat['quarter_start'])
        )
        
        if MIN_DAYS_THRESHOLD is not None and mask.any():
            overlap_start = compustat.loc[mask, 'quarter_start'].clip(lower=start_date)
            overlap_end = compustat.loc[mask, 'quarter_end'].clip(upper=end_date)
            overlap_days = (overlap_end - overlap_start).dt.days + 1
            
            mask_with_threshold = mask.copy()
            mask_with_threshold.loc[mask] = overlap_days >= MIN_DAYS_THRESHOLD
            compustat.loc[mask_with_threshold, 'in_sp500'] = True
        else:
            compustat.loc[mask, 'in_sp500'] = True

compustat = compustat.drop(columns=['quarter_start'])

compustat = compustat[compustat['in_sp500']].drop(columns=['in_sp500'])

compustat = compustat.drop(columns=["permco", "gvkey", "quarter",
                                     "fiscal_quarter", "linktype", "linkprim"])

db.close()