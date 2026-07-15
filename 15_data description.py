import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# Data description

# Preparation
SUBSAMPLE_END = "2015-09-30"
FULL_END      = "2023-12-31"
LTG_NW_LAGS   = 16
RED           = "firebrick"
BLUE          = "steelblue"

price_ratios_t2 = price_ratios.copy(deep=True)
eps_combined_t2 = eps_combined.copy(deep=True)
dps_combined_t2 = dps_combined.copy(deep=True)
eps_ltg_agg_t2  = eps_ltg_agg.copy(deep=True)
dps_ltg_agg_t2  = dps_ltg_agg.copy(deep=True)
gh_return_exp_t2 = gh_return_exp.copy(deep=True)

for _df, _col in [
    (price_ratios_t2, "quarter_end"),
    (eps_combined_t2, "estimate_date"),
    (dps_combined_t2, "estimate_date"),
    (eps_ltg_agg_t2, "estimate_date"),
    (dps_ltg_agg_t2, "estimate_date"),
]:
    if _col in _df.columns:
        _df[_col] = pd.to_datetime(_df[_col]).dt.normalize()

# Further preparation: OLS slope + Newey-West SE
def ols_slope_nw_se(y, x, lags=None):
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    T  = len(df)
    if T < 5:
        return (np.nan, np.nan)

    y_arr = df["y"].values
    x_arr = df["x"].values

    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lags = min(lags, T - 2)

    X   = np.column_stack([np.ones(T), x_arr])
    res = OLS(y_arr, X).fit()

    try:
        rob = res.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
        return float(rob.params[1]), float(rob.bse[1])
    except TypeError:
        V = cov_hac(res, nlags=lags, use_correction=True)
        return float(res.params[1]), float(np.sqrt(V[1, 1]))

def to_qlabel(ts):
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"

def compute_row(label, horizon, df_growth, date_col, growth_col,
                price_df, start, end,
                want_pd=True, want_pe=True, nw_lags=None):
    tmp = df_growth[[date_col, growth_col]].dropna().copy()
    tmp = tmp.rename(columns={date_col: "quarter_end", growth_col: "g"})
    tmp["quarter_end"] = pd.to_datetime(tmp["quarter_end"]).dt.normalize()
    tmp = tmp[(tmp["quarter_end"] >= pd.Timestamp(start)) &
              (tmp["quarter_end"] <= pd.Timestamp(end))]

    m = tmp.merge(price_df, on="quarter_end", how="inner").dropna(subset=["g"])
    T = len(m)

    empty = dict(label=label, horizon=horizon, n=T,
                 sample_start=start[:7], sample_end=end[:7],
                 std_dev=np.nan,
                 b_pd=np.nan, se_pd=np.nan, corr_pd=np.nan,
                 b_pe=np.nan, se_pe=np.nan, corr_pe=np.nan,
                 nw_lags_used=nw_lags, is_sub=False)
    if T < 5:
        return empty

    std_dev = m["g"].std(ddof=1) * 100

    b_pd, se_pd, b_pe, se_pe = np.nan, np.nan, np.nan, np.nan
    corr_pd, corr_pe = np.nan, np.nan

    if want_pd:
        b_pd, se_pd = ols_slope_nw_se(m["g"], m["log_pd"], lags=nw_lags)
        valid = m[["g", "log_pd"]].dropna()
        if len(valid) >= 5:
            corr_pd = float(valid["g"].corr(valid["log_pd"]))

    if want_pe:
        b_pe, se_pe = ols_slope_nw_se(m["g"], m["log_pe"], lags=nw_lags)
        valid = m[["g", "log_pe"]].dropna()
        if len(valid) >= 5:
            corr_pe = float(valid["g"].corr(valid["log_pe"]))

    actual_lags = (nw_lags if nw_lags is not None
                   else int(np.floor(4 * (T / 100) ** (2 / 9))))

    return dict(label=label, horizon=horizon, n=T,
                sample_start=to_qlabel(m["quarter_end"].min()),
                sample_end=to_qlabel(m["quarter_end"].max()),
                std_dev=std_dev,
                b_pd=b_pd, se_pd=se_pd, corr_pd=corr_pd,
                b_pe=b_pe, se_pe=se_pe, corr_pe=corr_pe,
                nw_lags_used=actual_lags, is_sub=False)


def row_pair(label, horizon, df, date_col, gcol,
             data_start, want_pd, want_pe, nw_lags=None):
    full = compute_row(label, horizon, df, date_col, gcol,
                       pr, data_start, FULL_END, want_pd, want_pe, nw_lags)
    sub  = compute_row(label, horizon, df, date_col, gcol,
                       pr, data_start, SUBSAMPLE_END, want_pd, want_pe, nw_lags)
    full["is_sub"] = False
    sub["is_sub"]  = True
    return [full, sub]


# Price Ratios
pr = price_ratios_t2[["quarter_end", "log_pd", "log_pe"]].copy()
pr["quarter_end"] = pd.to_datetime(pr["quarter_end"]).dt.normalize()
pr = pr.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_pd", "log_pe"])

# Build annualized 2yr growth series (based on De La O and Myers 2021)
eps_merge = (
    eps_combined_t2[["estimate_date", "ear_growth_1yr", "ear_growth_2yr"]]
    .dropna(subset=["ear_growth_1yr", "ear_growth_2yr"])
    .copy()
)
eps_merge["ear_growth_2yr_ann"] = (
    eps_merge["ear_growth_1yr"] + eps_merge["ear_growth_2yr"]
) / 2.0

dps_merge = (
    dps_combined_t2[["estimate_date", "div_growth_1yr", "div_growth_2yr"]]
    .dropna(subset=["div_growth_1yr", "div_growth_2yr"])
    .copy()
)
dps_merge["div_growth_2yr_ann"] = (
    dps_merge["div_growth_1yr"] + dps_merge["div_growth_2yr"]
) / 2.0

for name, df1yr, col1, df2yr, col2 in [
    ("EPS", eps_combined_t2, "ear_growth_1yr", eps_merge, "ear_growth_2yr_ann"),
    ("DPS", dps_combined_t2, "div_growth_1yr", dps_merge, "div_growth_2yr_ann"),
]:
    s1 = df1yr[col1].dropna()
    s2 = df2yr[col2].dropna()

# Prepare all series
dps1 = (dps_combined_t2[["estimate_date", "div_growth_1yr"]].dropna().copy())
eps1 = (eps_combined_t2[["estimate_date", "ear_growth_1yr"]].dropna().copy())

dps2_ann = dps_merge[["estimate_date", "div_growth_2yr_ann"]].dropna().copy()
eps2_ann = eps_merge[["estimate_date", "ear_growth_2yr_ann"]].dropna().copy()

dps_ltg = dps_ltg_agg_t2[["estimate_date", "forecast_ltg"]].dropna().copy()
eps_ltg = eps_ltg_agg_t2[["estimate_date", "forecast_ltg"]].dropna().copy()

gh = gh_return_exp_t2[["exp_ret_1y", "exp_ret_10y"]].copy()
gh.index = pd.to_datetime(gh.index).normalize()
gh = gh.reset_index()
gh.columns = ["quarter_end"] + list(gh.columns[1:])
gh["quarter_end"] = pd.to_datetime(gh["quarter_end"]).dt.normalize()

GH_START = "2001-10-01"

# Panel A: Dividend Growth
panel_a = []
panel_a += row_pair("I/B/E/S 1yr", "1",
                    dps1, "estimate_date", "div_growth_1yr",
                    "2003-01-01", want_pd=True, want_pe=False)
panel_a += row_pair("I/B/E/S 2yr", "2",
                    dps2_ann, "estimate_date", "div_growth_2yr_ann",
                    "2003-01-01", want_pd=True, want_pe=False)
panel_a += row_pair("I/B/E/S LTG", "LTG",
                    dps_ltg, "estimate_date", "forecast_ltg",
                    "2003-01-01", want_pd=True, want_pe=False,
                    nw_lags=LTG_NW_LAGS)

# Panel B: Earnings Growth
panel_b = []
panel_b += row_pair("I/B/E/S 1yr", "1",
                    eps1, "estimate_date", "ear_growth_1yr",
                    "1976-01-01", want_pd=False, want_pe=True)
panel_b += row_pair("I/B/E/S 2yr", "2",
                    eps2_ann, "estimate_date", "ear_growth_2yr_ann",
                    "1985-01-01", want_pd=False, want_pe=True)
panel_b += row_pair("I/B/E/S LTG", "LTG",
                    eps_ltg, "estimate_date", "forecast_ltg",
                    "1976-01-01", want_pd=False, want_pe=True,
                    nw_lags=LTG_NW_LAGS)

# Panel C: Return Expectations (G-H only)
panel_c = []
panel_c += row_pair("G-H", "1",
                    gh, "quarter_end", "exp_ret_1y",
                    GH_START, want_pd=True, want_pe=True)
panel_c += row_pair("G-H", "10",
                    gh, "quarter_end", "exp_ret_10y",
                    GH_START, want_pd=True, want_pe=True)


# Print Table
def fc(val, se, d=4):
    if pd.isna(val):
        return "", ""
    fmt = f"{{:.{d}f}}"
    return fmt.format(val), f"({fmt.format(se)})"

def fc_corr(val, d=3):
    if pd.isna(val):
        return ""
    return f"{val:.{d}f}"

def print_panel(panel_name, rows, col4_hdr, col5_hdr, col4c_hdr, col5c_hdr):
    print(f"\n{panel_name}")
    print(f"  {'Series':<16} {'Hor':>4}  {'Sample':>22}  {'Std%':>7}  "
          f"  {col4_hdr:>16} {'SE':>7} {'Corr':>6}"
          f"   {col5_hdr:>16} {'SE':>7} {'Corr':>6}"
          f"  {'N':>5}  {'NW lags':>7}")
    print("  " + "-" * 128)

    i = 0
    while i < len(rows):
        r_full = rows[i]
        r_sub  = rows[i + 1] if (i + 1) < len(rows) else None

        for r in [r_full, r_sub]:
            if r is None:
                continue
            c4, s4  = fc(r["b_pd"],  r["se_pd"])
            c5, s5  = fc(r["b_pe"],  r["se_pe"])
            cc4     = fc_corr(r["corr_pd"])
            cc5     = fc_corr(r["corr_pe"])
            sd_str  = f"{r['std_dev']:.1f}%" if not pd.isna(r["std_dev"]) else "--"
            smp     = f"{r['sample_start']} - {r['sample_end']}"
            lbl     = r["label"] if not r["is_sub"] else ""
            lag_str = str(r["nw_lags_used"]) if r["nw_lags_used"] is not None else "auto"

            print(f"  {lbl:<16} {r['horizon']:>4}  {smp:>22}  {sd_str:>7}  "
                  f"  {c4:>16} {s4:>7} {cc4:>6}"
                  f"   {c5:>16} {s5:>7} {cc5:>6}"
                  f"  {r['n']:>5}  {lag_str:>7}")

        if (i + 2) < len(rows) and rows[i]["label"] != rows[i + 2]["label"]:
            print()
        i += 2

SEP = "=" * 128

print("\n\n")
print(SEP)
print(" Comovement of Subjective Expectations and Price Ratios")
print("De La O & Myers (2021) - Full sample (->2023Q4) + paper subsample (->2015Q3)")
print(SEP)

print_panel("Panel A: Dividend Growth Expectations",
            panel_a,
            "cov(pd,g)/var(pd)", "[not reported]",
            "corr(pd,g)", "")
print_panel("Panel B: Earnings Growth Expectations",
            panel_b,
            "[not reported]", "cov(pe,g)/var(pe)",
            "", "corr(pe,g)")
print_panel("Panel C: Return Expectations (G-H CFO survey)",
            panel_c,
            "cov(pd,g)/var(pd)", "cov(pe,g)/var(pe)",
            "corr(pd,g)", "corr(pe,g)")

# Additional Diagnostic: Panel B Earnings - 2003-Based Windows
print("\n\n")
print(SEP)
print("Additional Diagnostic: Panel B Earnings - 2003Q1-Based Windows")
print(SEP)

diag_b = []
diag_b += row_pair("I/B/E/S 1yr", "1",
                   eps1, "estimate_date", "ear_growth_1yr",
                   "2003-01-01", want_pd=False, want_pe=True)
diag_b += row_pair("I/B/E/S 2yr", "2",
                   eps2_ann, "estimate_date", "ear_growth_2yr_ann",
                   "2003-01-01", want_pd=False, want_pe=True)
diag_b += row_pair("I/B/E/S LTG", "LTG",
                   eps_ltg, "estimate_date", "forecast_ltg",
                   "2003-01-01", want_pd=False, want_pe=True,
                   nw_lags=LTG_NW_LAGS)

print_panel("Panel B (2003Q1 base): Earnings Growth Expectations",
            diag_b,
            "[not reported]", "cov(pe,g)/var(pe)",
            "", "corr(pe,g)")

# Figures
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import math

mpl.rcParams.update({
    "font.size": 11,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 600,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "path.simplify": False,
    "agg.path.chunksize": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

YLABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13
TICK_FONTSIZE   = 11
LW_LTG  = 1.8
LW_RHS  = 1.6
ALPHA   = 1.0
DPS_LTG_END = pd.Timestamp("2015-09-30")

def make_five_year_ticks(ax, start_year: int, last_date: pd.Timestamp):
    five_yr_ticks = pd.date_range(start=f"{start_year}-01-01", end=last_date, freq="5YS")
    ax.set_xticks(five_yr_ticks)
    ax.set_xticklabels([t.strftime("%Y") for t in five_yr_ticks])
    ax.tick_params(axis="x", rotation=0, labelsize=TICK_FONTSIZE)

def style_axes(ax_left, ax_right, legend_loc):
    ax_left.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_right.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_left.grid(False)
    ax_right.grid(False)
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2,
                   loc=legend_loc, fontsize=LEGEND_FONTSIZE, frameon=False)
    ax_right.axhline(0, color="grey", linewidth=0.5, linestyle="-", alpha=0.5)

def plot_ltg_figure(df, date_col, ltg_col, rhs1_col, rhs2_col,
                    rhs_ylabel, legend_loc, xtick_start_year, out_file,
                    ltg_end=None, color=RED):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ltg_series = df[ltg_col].copy()
    if ltg_end is not None:
        ltg_series = ltg_series.where(df[date_col] <= ltg_end, np.nan)

    ax1.plot(
        df[date_col], ltg_series,
        color=color, linewidth=LW_LTG, linestyle="-", alpha=ALPHA, label=r"$\mathrm{LTG}_t$",
        antialiased=True, solid_capstyle="butt", solid_joinstyle="miter"
    )
    ax1.set_ylabel(r"$\mathrm{LTG}_t$", fontsize=YLABEL_FONTSIZE)

    ax2 = ax1.twinx()
    ax2.plot(
        df[date_col], df[rhs1_col],
        color=color, linewidth=LW_RHS, linestyle=":", alpha=ALPHA,
        label=df.attrs["rhs1_label"],
        antialiased=True, dash_capstyle="butt", dash_joinstyle="miter"
    )
    ax2.plot(
        df[date_col], df[rhs2_col],
        color=color, linewidth=LW_RHS, linestyle="--", alpha=ALPHA,
        label=df.attrs["rhs2_label"],
        antialiased=True, dash_capstyle="butt", dash_joinstyle="miter"
    )
    ax2.set_ylabel(rhs_ylabel, fontsize=YLABEL_FONTSIZE)

    first_date = df[date_col].min()
    last_date  = df[date_col].max()
    padding = pd.DateOffset(months=6)
    ax1.set_xlim(first_date - padding, last_date + padding)

    make_five_year_ticks(ax1, xtick_start_year, last_date)
    style_axes(ax1, ax2, legend_loc)

    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=600)

    plt.show()

# Figure 1 - Earnings
eps_plot = eps_ltg_agg[["estimate_date", "forecast_ltg"]].merge(
    eps_combined[["estimate_date", "ear_growth_1yr", "ear_growth_2yr"]],
    on="estimate_date",
    how="inner"
).sort_values("estimate_date").dropna()

eps_plot.attrs["rhs1_label"] = r"$\mathrm{E}_t^*[\Delta e_{t+1}]$"
eps_plot.attrs["rhs2_label"] = r"$\mathrm{E}_t^*[\Delta e_{t+2}]$"

print(f"EPS plot: {len(eps_plot)} obs, "
      f"{eps_plot['estimate_date'].min().date()} to "
      f"{eps_plot['estimate_date'].max().date()}")

plot_ltg_figure(
    df=eps_plot,
    date_col="estimate_date",
    ltg_col="forecast_ltg",
    rhs1_col="ear_growth_1yr",
    rhs2_col="ear_growth_2yr",
    rhs_ylabel=r"$\mathrm{E}_t^*[\Delta e_{t+1}]$, $\mathrm{E}_t^*[\Delta e_{t+2}]$",
    legend_loc="upper left",
    xtick_start_year=1985,
    out_file="figure1_earnings_expectations.png",
    color=BLUE
)

# Figure 2 - Dividends 
dps_rhs_full = dps_combined[["estimate_date", "div_growth_1yr", "div_growth_2yr"]].copy()
dps_ltg = dps_ltg_agg[["estimate_date", "forecast_ltg"]].copy()

dps_plot = dps_rhs_full.merge(
    dps_ltg,
    on="estimate_date",
    how="left"
).sort_values("estimate_date")

dps_plot = dps_plot.dropna(subset=["div_growth_1yr", "div_growth_2yr"]).copy()

dps_plot.attrs["rhs1_label"] = r"$\mathrm{E}_t^*[\Delta d_{t+1}]$"
dps_plot.attrs["rhs2_label"] = r"$\mathrm{E}_t^*[\Delta d_{t+2}]$"

print(f"DPS plot (RHS full): {len(dps_plot)} obs, "
      f"{dps_plot['estimate_date'].min().date()} to "
      f"{dps_plot['estimate_date'].max().date()}")
print(f"  Note: DPS LTG line shown only through {DPS_LTG_END.date()} (NaN afterward)")

plot_ltg_figure(
    df=dps_plot,
    date_col="estimate_date",
    ltg_col="forecast_ltg",
    rhs1_col="div_growth_1yr",
    rhs2_col="div_growth_2yr",
    rhs_ylabel=r"$\mathrm{E}_t^*[\Delta d_{t+1}]$, $\mathrm{E}_t^*[\Delta d_{t+2}]$",
    legend_loc="upper right",
    xtick_start_year=2005,
    out_file="figure2_dividends_expectations.png",
    ltg_end=DPS_LTG_END,
    color=RED
)

# Appendix Figures
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

save_dir = Path.cwd() / "figures"
save_dir.mkdir(parents=True, exist_ok=True)

spy_shiller = spy_quarterly.merge(
    shiller_quarterly[['quarter_end', 'D']],
    on='quarter_end',
    how='inner'
).dropna(subset=['spy_dividend_4q', 'D'])

spy_scale = spy_shiller['D'].mean() / spy_shiller['spy_dividend_4q'].mean()
print(f"SPY scaling factor: {spy_scale:.4f}")

spy_quarterly['spy_dividend_index'] = spy_quarterly['spy_dividend_4q'] * spy_scale

div_plot = (
    all_comp_agg[["quarter_end", "dividend_index"]]
    .merge(ibes_sp500_dps[["quarter_end", "dividends_index"]], on="quarter_end", how="outer")
    .merge(shiller_quarterly[["quarter_end", "D"]],            on="quarter_end", how="outer")
    .merge(spy_quarterly[["quarter_end", "spy_dividend_index"]], on="quarter_end", how="outer")
    .sort_values("quarter_end")
    .reset_index(drop=True)
)

earn_plot = (
    all_comp_agg[["quarter_end", "earnings_index"]]
    .merge(
        ibes_sp500_eps[["quarter_end", "earnings_index"]],
        on="quarter_end",
        how="outer",
        suffixes=("_all", "_ibes")
    )
    .merge(shiller_quarterly[["quarter_end", "E"]], on="quarter_end", how="outer")
    .sort_values("quarter_end")
    .reset_index(drop=True)
)

# Figure A1: Dividend Index
div_start = pd.Timestamp('2004-01-01')
div_end   = pd.Timestamp('2023-09-30')

div_plot_filtered = div_plot[
    (div_plot['quarter_end'] >= div_start) &
    (div_plot['quarter_end'] <= div_end)
].copy()

legend_handles = [
    plt.Line2D([0], [0], color='steelblue',  linewidth=1.2, linestyle='--', label='All Companies'),
    plt.Line2D([0], [0], color='firebrick',  linewidth=1.2, linestyle='-.', label='I/B/E/S'),
    plt.Line2D([0], [0], color='darkorange', linewidth=1.6, linestyle='-',  label='Shiller'),
    plt.Line2D([0], [0], color='purple',     linewidth=1.2, linestyle=':',  label='SPY'),
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(div_plot_filtered['quarter_end'], div_plot_filtered['dividend_index'],
        color='steelblue',  linewidth=1.2, linestyle='--')
ax.plot(div_plot_filtered['quarter_end'], div_plot_filtered['dividends_index'],
        color='firebrick',  linewidth=1.2, linestyle='-.')
ax.plot(div_plot_filtered['quarter_end'], div_plot_filtered['D'],
        color='darkorange', linewidth=1.6, linestyle='-')
ax.plot(div_plot_filtered['quarter_end'], div_plot_filtered['spy_dividend_index'],
        color='purple',     linewidth=1.2, linestyle=':')
ax.set_xlim(div_start, div_end)
ax.legend(handles=legend_handles, loc='upper left', fontsize=12, frameon=False)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(div_start - pd.DateOffset(months=3), div_end + pd.DateOffset(months=3))
ax.grid(False)
plt.tight_layout()
out1 = save_dir / "figure_a1_dividend_index.png"
fig.savefig(out1, dpi=1000, bbox_inches="tight")
print("Saved:", out1.resolve())
plt.show()
plt.close(fig)

# Figure A2: Earnings Index
earn_start = pd.Timestamp('1977-01-01')
earn_end   = pd.Timestamp('2023-09-30')

earn_plot_filtered = earn_plot[
    (earn_plot['quarter_end'] >= earn_start) &
    (earn_plot['quarter_end'] <= earn_end)
].copy()

legend_handles_earn = [
    plt.Line2D([0], [0], color='steelblue',  linewidth=1.2, linestyle='--', label='All Companies'),
    plt.Line2D([0], [0], color='firebrick',  linewidth=1.2, linestyle='-.', label='I/B/E/S'),
    plt.Line2D([0], [0], color='darkorange', linewidth=1.6, linestyle='-',  label='Shiller'),
]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(earn_plot_filtered['quarter_end'], earn_plot_filtered['earnings_index_all'],
        color='steelblue',  linewidth=1.2, linestyle='--')
ax.plot(earn_plot_filtered['quarter_end'], earn_plot_filtered['earnings_index_ibes'],
        color='firebrick',  linewidth=1.2, linestyle='-.')
ax.plot(earn_plot_filtered['quarter_end'], earn_plot_filtered['E'],
        color='darkorange', linewidth=1.6, linestyle='-')
ax.set_xlim(earn_start, earn_end)
ax.set_xlim(earn_start - pd.DateOffset(months=3), earn_end + pd.DateOffset(months=3))
ax.legend(handles=legend_handles_earn, loc='upper left', fontsize=12, frameon=False)
ax.tick_params(axis='both', labelsize=12)
ax.grid(False)
plt.tight_layout()
out2 = save_dir / "figure_a2_earnings_index.png"
fig.savefig(out2, dpi=1000, bbox_inches="tight")
print("Saved:", out2.resolve())
plt.show()
plt.close(fig)