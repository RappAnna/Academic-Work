import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# DPS coverage in IBES

# Prepare data
dps_1yr_cov = (
    dps_1_index[["estimate_date", "num_firms"]]
    .rename(columns={"num_firms": "n_firms"})
    .copy()
)
dps_2yr_cov = (
    dps_2_index[["estimate_date", "num_firms"]]
    .rename(columns={"num_firms": "n_firms"})
    .copy()
)
dps_ltg_cov = (
    dps_ltg_data[dps_ltg_data["forecast"].notna()]
    .groupby("estimate_date")["permno"]
    .nunique()
    .reset_index()
    .rename(columns={"permno": "n_firms"})
)
eps_ltg_cov = (
    eps_ltg_data[eps_ltg_data["forecast"].notna()]
    .groupby("estimate_date")["permno"]
    .nunique()
    .reset_index()
    .rename(columns={"permno": "n_firms"})
)

for df in [dps_1yr_cov, dps_2yr_cov, dps_ltg_cov, eps_ltg_cov]:
    df["estimate_date"] = pd.to_datetime(df["estimate_date"])

full_quarters = pd.DataFrame({
    "estimate_date": pd.date_range("1980-01-01", "2024-12-31", freq="Q")
})

def reindex_to_grid(cov_df, grid=full_quarters):
    return (
        grid
        .merge(cov_df, on="estimate_date", how="left")
        .sort_values("estimate_date")
    )

eps_ltg_grid  = reindex_to_grid(eps_ltg_cov)
dps_1yr_grid  = reindex_to_grid(dps_1yr_cov)
dps_2yr_grid  = reindex_to_grid(dps_2yr_cov)
dps_ltg_grid  = reindex_to_grid(dps_ltg_cov)

# Plot
BLUE_REF = "steelblue"
RED      = "firebrick"
LW       = 1.5

fig, ax = plt.subplots(figsize=(12, 4.5))

# EPS LTG reference (blue solid)
ax.plot(eps_ltg_grid["estimate_date"], eps_ltg_grid["n_firms"],
        color=BLUE_REF, linewidth=LW, linestyle="-", label="EPS LTG (reference)")

# DPS 1-year (red solid)
ax.plot(dps_1yr_grid["estimate_date"], dps_1yr_grid["n_firms"],
        color=RED, linewidth=LW, linestyle="-", label="DPS 1-year")

# DPS 2-year (red dashed)
ax.plot(dps_2yr_grid["estimate_date"], dps_2yr_grid["n_firms"],
        color=RED, linewidth=LW, linestyle="--", label="DPS 2-year")

# DPS LTG (red dotted)
ax.plot(dps_ltg_grid["estimate_date"], dps_ltg_grid["n_firms"],
        color=RED, linewidth=LW, linestyle=":", label="DPS LTG")

# Shade post-2015 DPS LTG collapse
last_date = dps_1yr_grid["estimate_date"].max()
ax.axvspan(pd.Timestamp("2015-07-01"), last_date,
           alpha=0.08, color="firebrick",
           label="DPS LTG coverage collapse (post-2015)")
ax.axvline(pd.Timestamp("2015-07-01"),
           color="firebrick", linewidth=1.0, linestyle=":", alpha=0.7)

ax.set_ylabel("Number of firms")
ax.legend(frameon=False, fontsize=12, loc="lower left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
ax.set_axisbelow(True)
ax.set_xlim(pd.Timestamp("1980-01-01"), last_date + pd.DateOffset(months=6))

plt.tight_layout()
plt.savefig("figure_dps_coverage_with_eps_ref.png", dpi=1000, bbox_inches="tight")
plt.show()