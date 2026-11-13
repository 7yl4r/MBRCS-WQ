"""Shared loading and plotting helpers for the MBeRCS / SFER exploration site.

The raw CSV is expected one directory up from this file, at
``../data/SFER_data.csv`` (i.e. ``<repo>/data/SFER_data.csv``).  See README.md
for where to download it.

Import this module from any page::

    import sfer_common as sfer
    df = sfer.load()
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.signal import lombscargle

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "SFER_data.csv"
LAND_PATH = HERE / "assets" / "florida-land.geojson"

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

# analyte column -> (label, units, group)
PARAMS: dict[str, tuple[str, str, str]] = {
    "temp": ("Temperature", "\u00b0C", "physical"),
    "sal": ("Salinity", "PSU", "physical"),
    "o2_ctd": ("Dissolved oxygen (CTD)", "mg L\u207b\u00b9", "oxygen"),
    "o2_disc": ("Dissolved oxygen (Winkler)", "\u00b5mol kg\u207b\u00b9", "oxygen"),
    "nh4": ("Ammonium (NH\u2084\u207a)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "po4": ("Phosphate (PO\u2084\u00b3\u207b)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "no3_no2": ("Nitrate + nitrite (NO\u2083\u207b+NO\u2082\u207b)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "no3": ("Nitrate (NO\u2083\u207b)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "no2": ("Nitrite (NO\u2082\u207b)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "si": ("Silicate (SiO\u2084)", "\u00b5mol L\u207b\u00b9", "nutrient"),
    "chl": ("Chlorophyll a", "\u00b5g L\u207b\u00b9", "biological"),
    "pheo": ("Pheophytin", "\u00b5g L\u207b\u00b9", "biological"),
}

# Every analyte, in the order they are presented on the site.  The pages loop
# over this list so that no analyte is silently dropped from a figure or table.
ANALYTES = list(PARAMS)

# Compact labels, for axes with little room (e.g. the coverage heatmap).
SHORT_LABELS = {
    "temp": "Temp", "sal": "Salinity", "o2_ctd": "O\u2082 (CTD)",
    "o2_disc": "O\u2082 (Winkler)", "nh4": "NH\u2084", "po4": "PO\u2084",
    "no3_no2": "NO\u2083+NO\u2082", "no3": "NO\u2083", "no2": "NO\u2082",
    "si": "SiO\u2084", "chl": "Chl a", "pheo": "Pheo",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DEPTH_ORDER = ["Surface", "Mid", "Bottom"]

# Vessel codes that appear as the prefix of ``cruise_id``.
VESSELS = {"WS": "Weatherbird II (WS)", "HG": "Hogarth (HG)",
           "WB": "WB (WB)", "SV": "SV (SV)"}

# Columns that should stay as text (everything else is coerced to numbers).
_TEXT_COLS = {"keyfield", "cruise_id", "time", "datetime", "station",
              "station_type", "depth_class", "depth_order", "notes"}


def label(col: str) -> str:
    """Human-readable label for an analyte column."""
    return PARAMS.get(col, (col, "", ""))[0]


def units(col: str) -> str:
    return PARAMS.get(col, ("", "", ""))[1]


def short(col: str) -> str:
    """Compact label, for axes with little room (e.g. the coverage heatmap)."""
    return SHORT_LABELS.get(col, label(col))


# Values outside these ranges are known data errors (e.g. a single CTD oxygen
# reading of ~6,000 mg/L).  They are kept in the tables but dropped from plots,
# where one impossible point would otherwise stretch an entire axis.
PLAUSIBLE = {"o2_ctd": (0.0, 15.0)}


def plausible(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask of rows with a present, physically plausible value for ``col``."""
    s = df[col]
    mask = s.notna()
    lo, hi = PLAUSIBLE.get(col, (None, None))
    if lo is not None:
        mask &= s.between(lo, hi)
    return mask


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    """Load and lightly clean the SFER discrete water-quality CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file: {DATA_PATH}\n"
            "Download SFER_data.csv from "
            "https://github.com/Ecosystem-Assessment-Lab/SFER/blob/main/DATA/SFER_data.csv "
            "and save it as ../data/SFER_data.csv (see README.md)."
        )

    df = pd.read_csv(DATA_PATH)

    # Coerce every non-text column to numeric.  Some columns (e.g. o2_ctd)
    # contain placeholder strings such as "Under Review" that become NaN.
    for col in df.columns:
        if col not in _TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["vessel"] = df["cruise_id"].str.slice(0, 2).map(VESSELS).fillna(df["cruise_id"].str.slice(0, 2))
    df["month_name"] = df["month"].map(lambda m: MONTHS[int(m) - 1] if pd.notna(m) else np.nan)
    df["season"] = df["month"].map(_season)
    df["depth_class"] = pd.Categorical(df["depth_class"], categories=DEPTH_ORDER, ordered=True)
    return df


def _season(month) -> str:
    if pd.isna(month):
        return np.nan
    m = int(month)
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def available(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask of rows with a numeric value for ``col``."""
    return df[col].notna()


def stats_table(df: pd.DataFrame, cols=None, depth_class: str | None = None) -> pd.DataFrame:
    """Basic descriptive statistics for each analyte (optionally one depth class)."""
    cols = cols or ANALYTES
    sub = df if depth_class is None else df[df["depth_class"] == depth_class]
    rows = []
    for col in cols:
        s = sub[col].dropna()
        if s.empty:
            continue
        rows.append({
            "Analyte": label(col),
            "Units": units(col),
            "n": int(s.size),
            "Min": s.min(),
            "Q25": s.quantile(0.25),
            "Median": s.median(),
            "Mean": s.mean(),
            "Q75": s.quantile(0.75),
            "Max": s.max(),
        })
    return pd.DataFrame(rows).set_index("Analyte")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def style() -> None:
    """Apply a consistent, publication-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


SEASON_COLORS = {"Winter": "#3b6fb6", "Spring": "#5aa469",
                 "Summer": "#e08214", "Autumn": "#b24745"}

DEPTH_COLORS = {"Surface": "#1f78b4", "Mid": "#7fbf7b", "Bottom": "#8c510a"}


def time_axis(ax, df, kind="date"):
    """Pin an x-axis to the full span of the record.

    Every dated panel on the site then shares the same time axis, so a short
    series (e.g. CTD oxygen, which only runs 2022-2026) sits on the same scale
    as the rest instead of being stretched to fill its own plot.  ``kind="year"``
    gives one integer tick per calendar year; ``kind="date"`` uses the exact
    first-to-last sample window.
    """
    if kind == "year":
        lo, hi = int(df["year"].min()), int(df["year"].max())
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_xticks(range(lo, hi + 1))
    else:
        ax.set_xlim(df["datetime"].min(), df["datetime"].max())
    return ax


def plot_basemap(ax, facecolor="#ece7dd", edgecolor="#b9b1a3", lw=0.6, zorder=0):
    """Draw the vendored Florida land polygons onto ``ax`` (no extra deps)."""
    if not LAND_PATH.exists():
        return
    with open(LAND_PATH) as fh:
        gj = json.load(fh)

    polys = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            polys.append(coords)
        elif gtype == "MultiPolygon":
            polys.extend(coords)

    patches = []
    for poly in polys:
        if not poly:
            continue
        exterior = np.asarray(poly[0])  # first ring is the exterior
        if exterior.ndim == 2 and len(exterior) >= 3:
            patches.append(MplPolygon(exterior, closed=True))
    if patches:
        ax.add_collection(PatchCollection(patches, facecolor=facecolor,
                                          edgecolor=edgecolor, linewidth=lw,
                                          zorder=zorder))


def violin_by_position(ax, data, positions=None, labels=None, logy=False,
                       colors="#8fbcd4", width=0.8, show_median=True):
    """Draw violin plots for a sequence of 1-D arrays.

    ``data`` is a sequence of arrays, one per group.  When ``logy`` is set the
    values are transformed to log10 *before* the density is estimated, so the
    violins keep their shape on a log axis, and the y ticks are relabelled as
    powers of ten.  ``colors`` may be a single colour or one colour per group.
    """
    data = [np.asarray(a, dtype=float) for a in data]
    if positions is None:
        positions = np.arange(1, len(data) + 1)

    kept = []
    for i, (pos, arr) in enumerate(zip(positions, data)):
        arr = arr[np.isfinite(arr)]
        if logy:
            arr = arr[arr > 0]
            arr = np.log10(arr)
        if arr.size >= 2:
            kept.append((i, pos, arr))
    if not kept:
        return

    pos = [p for _, p, _ in kept]
    arrs = [a for _, _, a in kept]
    if isinstance(colors, str):
        facecolors = [colors] * len(arrs)
    else:
        colors = list(colors)
        facecolors = [colors[i] for i, _, _ in kept]

    vp = ax.violinplot(arrs, positions=pos, widths=width,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, fc in zip(vp["bodies"], facecolors):
        body.set_facecolor(fc)
        body.set_edgecolor("#4a6f92")
        body.set_alpha(0.85)
        body.set_linewidth(0.8)

    if show_median:
        ax.scatter(pos, [np.median(a) for a in arrs], s=16, marker="o",
                   facecolor="white", edgecolor="#08306b", linewidths=1.0,
                   zorder=3)

    if logy:
        lo = min(a.min() for a in arrs)
        hi = max(a.max() for a in arrs)
        ticks = list(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$10^{{{t}}}$" for t in ticks])

    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)


def month_violin(df, col, ax, depth_class="Surface", logy=False, color="#8fbcd4"):
    """Violin plot of ``col`` by calendar month."""
    sub = df[(df["depth_class"] == depth_class) & plausible(df, col)]
    data = [sub.loc[sub["month"] == m, col].values for m in range(1, 13)]
    violin_by_position(ax, data, positions=list(range(1, 13)), labels=MONTHS,
                       logy=logy, colors=color)
    ax.set_ylabel(f"{label(col)} ({units(col)})")
    ax.set_title(label(col))


# ---------------------------------------------------------------------------
# Time series / seasonal decomposition
# ---------------------------------------------------------------------------

def monthly_series(df, col, depth_class="Surface", how="mean", interpolate=True):
    """Build a regular monthly time series of ``col``.

    The cruises are bi-monthly, so roughly half of the calendar months have no
    sample.  ``interpolate`` fills those gaps with time-based linear
    interpolation so the series is evenly spaced and can be decomposed.
    """
    sub = df[df[col].notna()].dropna(subset=["datetime"])
    if depth_class is not None:
        sub = sub[sub["depth_class"] == depth_class]
    s = sub.set_index("datetime")[col].resample("MS").agg(how)
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full)
    if interpolate:
        s = s.interpolate(method="time").ffill().bfill()
    return s.rename(col)


# Gaussian scale (in samples/months) of the trend smoother.  A Gaussian is used
# rather than a rectangular moving average because a moving average has hard
# notches at 12, 24, 36 ... months, which push genuine interannual variability
# out of the trend and into the residuals, making them look artificially
# periodic.  A Gaussian has a smooth frequency response with no such notch.
TREND_SIGMA = 9


def _gaussian_smooth(values, sigma, min_support_frac=0.85):
    """Edge-aware Gaussian smoother (numpy only, no scipy).

    Values are set to NaN where the available kernel weight falls below
    ``min_support_frac`` of the full kernel, so the unreliable one-sided
    estimates at the ends of the record are dropped.
    """
    x = np.asarray(values, float)
    n = x.size
    half = int(np.ceil(3 * sigma))
    offs = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (offs / sigma) ** 2)
    kern = kern / kern.sum()
    good = np.isfinite(x)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = kern[(half - (i - lo)):(half + (hi - i))] * good[lo:hi]
        total = w.sum()
        if total >= min_support_frac * kern.sum():
            out[i] = np.sum(x[lo:hi] * w) / total
    return out


def seasonal_decompose(series, period=12, trend_sigma=TREND_SIGMA):
    """Classical **additive** seasonal decomposition (no statsmodels needed).

    Returns a DataFrame with ``observed``, ``trend``, ``seasonal`` and ``resid``
    columns.  The trend is a Gaussian-weighted local average with scale
    ``trend_sigma`` months; the seasonal term is the mean detrended value for
    each position in the cycle, shifted so that the seasonal indices sum to
    zero; the residual is whatever the trend and seasonal terms do not explain.
    """
    s = pd.Series(series, dtype=float).astype(float)
    trend = pd.Series(_gaussian_smooth(s.to_numpy(), trend_sigma), index=s.index)

    detrended = s - trend
    if isinstance(s.index, pd.DatetimeIndex):
        pos = pd.Series(s.index.month, index=s.index)
    else:
        pos = pd.Series(np.arange(len(s)) % period, index=s.index)

    seasonal_means = detrended.groupby(pos).mean()
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal = pos.map(seasonal_means)
    seasonal.index = s.index

    resid = s - trend - seasonal
    return pd.DataFrame({"observed": s, "trend": trend,
                         "seasonal": seasonal, "resid": resid})


DECOMP_COLORS = {"trend": "#b24745", "seasonal": "#5aa469", "residual": "#8c510a"}


def decomposition_plot(df, col, ax, depth_class="Surface", period=12,
                       trend_sigma=TREND_SIGMA, series=None):
    """Draw an additive decomposition of ``col`` on a single axes.

    The trend is shown as an anomaly from the series mean, so the trend,
    seasonal and residual components share one scale.  Residuals are drawn as
    points with vertical stems down to the zero line, but **only at the months
    that actually have a sample**: the bi-monthly cruises leave roughly half the
    calendar months empty, and those are filled by interpolation to fit the
    trend and seasonal curves, not plotted as if they were observations.
    Returns the component DataFrame from :func:`seasonal_decompose`.
    """
    if series is not None:
        s = series
        observed = pd.Series(True, index=s.index)
    else:
        observed = monthly_series(df, col, depth_class, interpolate=False).notna()
        s = monthly_series(df, col, depth_class)
    d = seasonal_decompose(s, period=period, trend_sigma=trend_sigma)
    trend = d["trend"] - s.mean()
    seasonal = d["seasonal"]
    resid = d["resid"]

    real = observed.reindex(resid.index, fill_value=False).to_numpy()
    mask = np.isfinite(resid.to_numpy()) & real
    rx = resid.index[mask]
    rv = resid.to_numpy()[mask]

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.vlines(rx, 0, rv, color=DECOMP_COLORS["residual"], lw=0.7,
              alpha=0.55, zorder=1)
    ax.plot(trend.index, trend.to_numpy(), color=DECOMP_COLORS["trend"],
            lw=2.0, zorder=3, label="Trend")
    ax.plot(seasonal.index, seasonal.to_numpy(), color=DECOMP_COLORS["seasonal"],
            lw=1.3, zorder=4, label="Seasonal")
    ax.plot(rx, rv, "o", ms=2.5, color=DECOMP_COLORS["residual"],
            zorder=2, label="Residual")
    ax.set_title(label(col))
    ax.set_ylabel(f"Deviation from mean ({units(col)})")
    ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    return d


# ---------------------------------------------------------------------------
# Harmonic (Fourier) model of the seasonal cycle
# ---------------------------------------------------------------------------
#
# The classical decomposition above uses one fixed value per calendar month; it
# answers *how much* of a series is seasonal, not *what shape* the seasonality
# has.  The harmonic model below writes the detrended signal as a sum of
# sinusoids and ranks the harmonics by how much variance each one explains.

# Harmonics kept for the seasonal model.  The cruises sample roughly every two
# months, so the Nyquist period is about four months and harmonics shorter than
# that are aliased; they are not fitted.
HARMONIC_PERIODS = {1: 12.0, 2: 6.0, 3: 4.0}
HARMONIC_LABELS = {1: "Annual (12 mo)", 2: "Semi-annual (6 mo)",
                   3: "Four-monthly (4 mo)"}
HARMONIC_COLORS = {1: "#1f78b4", 2: "#e6821c", 3: "#6a9f58"}


def _harmonic_design(pos, n_harm):
    """Design matrix [1, cos(k t), sin(k t)] for k = 1 .. ``n_harm``."""
    cols = [np.ones_like(pos)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * pos / 12), np.sin(2 * np.pi * k * pos / 12)]
    return np.column_stack(cols)


def _detrended_observed(df, col, depth_class, trend_sigma):
    """Detrended monthly means using only months with real observations."""
    s = monthly_series(df, col, depth_class, interpolate=True)
    trend = seasonal_decompose(s, trend_sigma=trend_sigma)["trend"]
    obs = monthly_series(df, col, depth_class, interpolate=False)
    detr = (obs - trend).dropna()
    pos = np.array([s.index.get_loc(i) for i in detr.index], dtype=float)
    return s, pos, detr.to_numpy(dtype=float)


def harmonic_fit(df, col, depth_class="Surface", n_harm=3, trend_sigma=TREND_SIGMA):
    """Least-squares Fourier fit to the detrended monthly-mean series.

    Returns ``(table, info)``.  ``table`` has one row per harmonic with its
    period, amplitude, the share of the detrended variance it explains, the
    calendar month of its peak, and the OLS p-value.  ``info`` carries the
    coefficients, the origin month and the overall seasonal R².
    """
    s, pos, y = _detrended_observed(df, col, depth_class, trend_sigma)
    X = _harmonic_design(pos, n_harm)
    fit = sm.OLS(y, X).fit()
    beta = fit.params
    total_var = float(np.var(y))

    m0 = s.index[0].month
    tt = np.linspace(0, 12, 721, endpoint=False)
    rows = []
    for k in range(1, n_harm + 1):
        a, b = float(beta[2 * k - 1]), float(beta[2 * k])
        amp = float(np.hypot(a, b))
        comp = a * np.cos(2 * np.pi * k * tt / 12) + b * np.sin(2 * np.pi * k * tt / 12)
        peak = float(tt[int(np.argmax(comp))])
        month = int(((m0 - 1 + peak) % 12) + 1)
        rows.append({
            "harmonic": k,
            "Period (mo)": HARMONIC_PERIODS.get(k, 12.0 / k),
            "Amplitude": amp,
            "Share of variance": (0.5 * amp * amp / total_var) if total_var > 0 else np.nan,
            "Peak month": MONTHS[month - 1],
            "p": float(fit.pvalues[2 * k - 1]),
        })
    table = pd.DataFrame(rows)
    info = {
        "beta": beta,
        "n_harm": n_harm,
        "origin_month": m0,
        "total_var": total_var,
        "seasonal_r2": float(1 - np.var(y - fit.fittedvalues) / total_var)
        if total_var > 0 else np.nan,
        "n_months": len(y),
        "series": s,
    }
    return table, info


def _harmonic_component(beta, k, months, origin_month):
    """Evaluate harmonic ``k`` at a vector of calendar months (1-12)."""
    a, b = beta[2 * k - 1], beta[2 * k]
    phase = (np.asarray(months, dtype=float) - origin_month) % 12
    return a * np.cos(2 * np.pi * k * phase / 12) + b * np.sin(2 * np.pi * k * phase / 12)


def harmonic_plot(df, col, ax, depth_class="Surface", n_harm=3,
                  trend_sigma=TREND_SIGMA):
    """Draw the Fourier components of one analyte's seasonal cycle."""
    table, info = harmonic_fit(df, col, depth_class, n_harm, trend_sigma)
    beta, m0 = info["beta"], info["origin_month"]

    months = np.linspace(1, 13, 361)
    total = np.zeros_like(months)
    for k in range(1, n_harm + 1):
        comp = _harmonic_component(beta, k, months, m0)
        total = total + comp
        share = table.loc[k - 1, "Share of variance"] * 100
        ax.plot(months, comp, color=HARMONIC_COLORS.get(k, "#888888"), lw=1.7,
                label=f"{HARMONIC_LABELS.get(k, f'{12 / k:g} mo')} — {share:.0f}%")
    ax.plot(months, total, color="#333333", lw=2.2, label="Sum (seasonal)")

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlim(1, 13)
    ax.set_xticks(np.arange(1.5, 13.5, 1))
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel(f"Seasonal anomaly ({units(col)})")
    ax.set_title(f"{label(col)} — seasonal R² = {info['seasonal_r2']:.2f} "
                 f"(n = {info['n_months']} months)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return table, info


def harmonic_composition_plot(df, cols, ax, depth_class="Surface", n_harm=3):
    """Stacked horizontal bars: the harmonic make-up of each analyte."""
    records = []
    for col in cols:
        table, info = harmonic_fit(df, col, depth_class, n_harm)
        rec = {"col": col, "r2": info["seasonal_r2"]}
        for k in range(1, n_harm + 1):
            rec[k] = table.loc[k - 1, "Share of variance"] * 100
        records.append(rec)
    data = pd.DataFrame(records).sort_values("r2", ascending=True)

    ypos = np.arange(len(data))
    left = np.zeros(len(data))
    for k in range(1, n_harm + 1):
        ax.barh(ypos, data[k], left=left, height=0.72,
                color=HARMONIC_COLORS[k], label=HARMONIC_LABELS[k])
        left = left + data[k].to_numpy()

    ax.set_yticks(ypos)
    ax.set_yticklabels([label(c) for c in data["col"]])
    ax.set_xlabel("Share of detrended variance explained (%)")
    ax.set_title("Seasonal composition — bar length is the total seasonal R²")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, max(5.0, float(left.max()) * 1.05))
    return data


# ---------------------------------------------------------------------------
# Spectral view (Lomb-Scargle, for the irregularly sampled observations)
# ---------------------------------------------------------------------------

def lomb_scargle(df, col, depth_class="Surface", periods=None,
                 trend_sigma=TREND_SIGMA, pmin=3.0, pmax=48.0, nper=600):
    """Normalised Lomb-Scargle periodogram of the detrended observations."""
    _, pos, y = _detrended_observed(df, col, depth_class, trend_sigma)
    y = y - y.mean()
    if periods is None:
        periods = np.geomspace(pmin, pmax, nper)
    power = lombscargle(pos, y, 2 * np.pi / periods, normalize=True)
    return periods, power


def lomb_scargle_peaks(df, col, depth_class="Surface", n=3, pmin=3.0, pmax=48.0):
    """Top local maxima of the Lomb-Scargle periodogram as (period, power)."""
    periods, power = lomb_scargle(df, col, depth_class, pmin=pmin, pmax=pmax,
                                  nper=2000)
    idx = [i for i in range(1, len(power) - 1)
           if power[i] > power[i - 1] and power[i] >= power[i + 1]]
    idx = sorted(idx, key=lambda i: -power[i])[:n]
    return [(round(float(periods[i]), 1), round(float(power[i]), 3)) for i in idx]


def periodogram_heatmap(df, cols, ax, depth_class="Surface",
                        pmin=3.0, pmax=48.0, nper=300):
    """Heatmap of Lomb-Scargle power (analytes x period)."""
    periods = np.geomspace(pmin, pmax, nper)
    power = np.vstack([lomb_scargle(df, c, depth_class, periods=periods)[1]
                       for c in cols])
    ypos = np.arange(len(cols))
    mesh = ax.pcolormesh(periods, ypos, power, shading="auto", cmap="magma",
                         vmin=0, vmax=float(np.nanpercentile(power, 99)))
    ax.set_xscale("log")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([label(c) for c in cols])
    ax.set_xlabel("Period (months, log scale)")
    ax.set_title("Lomb–Scargle periodogram of the detrended observations")
    for p in (6.0, 12.0):
        ax.axvline(p, color="#7fd4ff", lw=0.8, ls="--", alpha=0.7)
    ax.figure.colorbar(mesh, ax=ax, label="Normalized power", shrink=0.85,
                       pad=0.02)
    return periods, power
