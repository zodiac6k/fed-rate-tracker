# streamlit_app.py — Fed Rate Cut Tracker & Macro Dashboard (no yfinance)
# ----------------------------------------------------------------------
# Data sources:
# - Fed press releases (Atom feed) + HTML fallback
# - Index/ETF: FRED CSV for SP500, Stooq CSV for SPY/QQQ
# - US recessions & macro: FRED CSV (no API key)
#
# Run locally:
#   pip install streamlit pandas numpy requests beautifulsoup4 plotly \
#               python-dateutil
#   streamlit run streamlit_app.py

import io
import re
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

# --- Config
st.set_page_config(
    page_title="Fed Rate Cut Tracker",
    page_icon="📉",
    layout="wide",
)

# Feeds & CSV endpoints
RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
STQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

# Classification patterns for statements
CUT_PATTERNS = [
    r"decided to lower the target range",
    r"lowered the federal funds rate",
    r"reduce the target range",
]
HIKE_PATTERNS = [
    r"decided to raise the target range",
    r"raised the federal funds rate",
    r"increase the target range",
]
HOLD_PATTERNS = [
    r"decided to maintain the target range",
    r"kept the target range",
    r"maintain its target range",
]
FIRST_CUT_COOLDOWN_MONTHS = 4

# Policy rate series (FRED)
RATE_SERIES = ["DFEDTARU", "DFEDTARL", "FEDFUNDS"]

# Polite User-Agent to avoid occasional 403s
REQUEST_KW = {
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; fed-rate-tracker/1.0; +streamlit)"
        )
    },
    "timeout": 20,
}

# -------------------------------
# Helpers — FRED & Stooq
# -------------------------------


@st.cache_data(ttl=3600)
def fred_csv(series_ids, start: str = "1980-01-01") -> pd.DataFrame:
    """
    Fetch one or more FRED series via fredgraph CSV.
    Returns a wide DataFrame indexed by date.
    """
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    params = {"id": ",".join(series_ids)}
    r = requests.get(FRED_CSV_BASE, params=params, **REQUEST_KW)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df[df["DATE"] >= pd.to_datetime(start)]
    df.set_index("DATE", inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def fetch_stooq_daily(
    symbol: str,
    start: str = "1985-01-01",
) -> pd.DataFrame:
    """
    Download daily OHLCV from Stooq, e.g. 'spy.us', 'qqq.us'.
    """
    url = STQ_DAILY_URL.format(symbol=symbol)
    r = requests.get(url, **REQUEST_KW)
    if r.status_code != 200 or not r.text or r.text.lower().startswith("html"):
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    ok_cols = "Date" in df.columns and "Close" in df.columns
    if df.empty or not ok_cols:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[df["Date"] >= pd.to_datetime(start)]
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df.set_index("date", inplace=True)
    return df[["close"]]


@st.cache_data(ttl=3600)
def fetch_prices(
    ticker: str,
    start: str = "1985-01-01",
) -> pd.DataFrame:
    """
    Unified price fetcher without yfinance.
    - ^GSPC → FRED 'SP500'
    - SPY/QQQ → Stooq 'spy.us' / 'qqq.us'
    """
    t = (ticker or "").strip()
    t_upper = t.upper()

    # S&P 500 via FRED
    if t_upper in {"^GSPC", "SPX", "^SPX", "S&P500", "SP500"}:
        try:
            df = fred_csv("SP500", start=start)
            if "SP500" in df.columns:
                out = df[["SP500"]].rename(columns={"SP500": "close"})
                return out
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    # Common ETFs via Stooq
    stooq_map = {
        "SPY": "spy.us",
        "QQQ": "qqq.us",
        "DIA": "dia.us",
        "IWM": "iwm.us",
    }
    sym = stooq_map.get(t_upper)
    if sym is None:
        t_lower = t.lower()
        sym = f"{t_lower}.us" if t_lower else None
    if sym:
        return fetch_stooq_daily(sym, start=start)

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_policy_rates(start: str) -> pd.DataFrame:
    """
    Returns daily policy rate frame:
    DFEDTARU (upper), DFEDTARL (lower), FEDFUNDS (effective).
    """
    try:
        df = fred_csv(RATE_SERIES, start=start).copy()
        return df
    except Exception:
        return pd.DataFrame()


def nearest_value(df: pd.DataFrame, col: str, dt: pd.Timestamp) -> float:
    """Nearest forward date's value for a given column."""
    if df.empty or col not in df.columns:
        return np.nan
    if dt in df.index:
        return float(df.loc[dt, col])
    idx = df.index.searchsorted(dt)
    if idx < len(df.index):
        return float(df.iloc[idx][col])
    return np.nan

# -------------------------------
# Fed press releases — Atom + fallback
# -------------------------------


@st.cache_data(ttl=3600)
def fetch_fed_press_releases() -> pd.DataFrame:
    rows = []

    # Atom feed first
    try:
        r = requests.get(RSS_URL, **REQUEST_KW)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for e in root.findall(".//atom:entry", ns):
            title = e.findtext("atom:title", namespaces=ns) or ""
            link_el = (
                e.find("atom:link[@rel='alternate']", ns) or
                e.find("atom:link", ns)
            )
            link = link_el.attrib.get("href") if link_el is not None else None
            published = (
                e.findtext("atom:published", namespaces=ns) or
                e.findtext("atom:updated", namespaces=ns) or
                ""
            )
            published_ts = pd.to_datetime(published, errors="coerce")
            if not link:
                continue
            try:
                html = requests.get(link, **REQUEST_KW).text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(" ", strip=True).lower()
            except Exception:
                text = title.lower()

            def _m(pats):
                return any(re.search(p, text) for p in pats)

            if _m(CUT_PATTERNS):
                action = "cut"
            elif _m(HIKE_PATTERNS):
                action = "hike"
            elif _m(HOLD_PATTERNS):
                action = "hold"
            else:
                action = "unknown"

            rows.append(
                {
                    "date": published_ts,
                    "action": action,
                    "title": title,
                    "url": link,
                }
            )
    except Exception as e:
        msg = (
            "RSS fetch failed "
            f"({e}); attempting HTML fallback…"
        )
        st.warning(msg)
        try:
            pr = requests.get(
                "https://www.federalreserve.gov/newsevents/pressreleases.htm",
                **REQUEST_KW,
            ).text
            soup = BeautifulSoup(pr, "html.parser")
            for a in soup.select("a"):
                at = (a.get_text(strip=True) or "").lower()
                href = a.get("href") or ""
                is_stmt = "fomc statement" in at
                starts_ok = href.startswith("/newsevents/pressreleases/")
                if is_stmt and starts_ok:
                    link = f"https://www.federalreserve.gov{href}"
                    page = requests.get(link, **REQUEST_KW).text
                    psoup = BeautifulSoup(page, "html.parser")
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", href)
                    dt = pd.NaT
                    if m:
                        y, mo, d = map(int, m.groups())
                        dt = pd.Timestamp(y, mo, d)
                    text = psoup.get_text(" ", strip=True).lower()

                    def _m2(pats):
                        return any(re.search(p, text) for p in pats)

                    if _m2(CUT_PATTERNS):
                        action = "cut"
                    elif _m2(HIKE_PATTERNS):
                        action = "hike"
                    elif _m2(HOLD_PATTERNS):
                        action = "hold"
                    else:
                        action = "unknown"

                    title = (
                        psoup.title.get_text(strip=True)
                        if psoup.title else
                        "FOMC statement"
                    )
                    rows.append(
                        {
                            "date": dt,
                            "action": action,
                            "title": title,
                            "url": link,
                        }
                    )
        except Exception as e2:
            st.error(f"Fallback scrape failed: {e2}")

    df = pd.DataFrame(rows, columns=["date", "action", "title", "url"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(
            drop=True
        )
        mask = df["title"].str.contains(
            (
                "FOMC statement|"
                "Federal Open Market Committee"
            ),
            case=False,
            na=False,
        )
        df = df.loc[mask].reset_index(drop=True)
    return df

# -------------------------------
# Return math
# -------------------------------


def nearest_close(df: pd.DataFrame, dt: pd.Timestamp) -> float:
    if df.empty:
        return np.nan
    if dt in df.index:
        if "close" in df.columns:
            return float(df.loc[dt, "close"])
        return float(df.loc[dt, "adj close"])
    idx = df.index.searchsorted(dt)
    if idx < len(df.index):
        row = df.iloc[idx]
        return float(row.get("close", row.get("adj close", np.nan)))
    return np.nan


def fwd_return(df: pd.DataFrame, start: pd.Timestamp, months: int) -> float:
    s = nearest_close(df, start)
    e = nearest_close(df, start + relativedelta(months=months))
    if np.isnan(s) or np.isnan(e):
        return np.nan
    return (e / s - 1) * 100


def identify_first_cuts(
    df: pd.DataFrame,
    cooldown_months: int = FIRST_CUT_COOLDOWN_MONTHS,
) -> pd.DataFrame:
    base_cols = ["date", "action", "title", "url"]
    if df.empty or "action" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=base_cols)
    cuts = df[df.action == "cut"].copy()
    if cuts.empty:
        return pd.DataFrame(columns=base_cols)

    cuts["prev"] = cuts["date"].shift(1)

    def _is_first(r):
        if pd.isna(r.prev):
            return True
        delta = r.date - r.prev
        return delta >= pd.DateOffset(months=cooldown_months)

    cuts["is_first"] = cuts.apply(_is_first, axis=1)
    out = cuts[cuts.is_first].drop(
        columns=["prev", "is_first"],
        errors="ignore",
    )
    return out.reset_index(drop=True)


def annotate_rates(
    events: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds rate_lower, rate_upper, rate_mid, eff_rate and delta_mid_bps
    from the prior first-cut event.
    """
    if events.empty or rates.empty:
        return events

    out = events.copy().sort_values("date").reset_index(drop=True)
    lows, ups, mids, effs = [], [], [], []
    for _, r in out.iterrows():
        d = r.date
        lo = nearest_value(rates, "DFEDTARL", d)
        up = nearest_value(rates, "DFEDTARU", d)
        eff = nearest_value(rates, "FEDFUNDS", d)
        mid = np.nanmean([lo, up]) if not np.isnan(lo + up) else np.nan
        lows.append(lo)
        ups.append(up)
        mids.append(mid)
        effs.append(eff)

    out["rate_lower"] = lows
    out["rate_upper"] = ups
    out["rate_mid"] = mids
    out["eff_rate"] = effs

    out["delta_mid_bps"] = (
        (out["rate_mid"].diff()) * 100 if out["rate_mid"].notna().any()
        else np.nan
    )
    return out


def compute_returns(
    first_cuts: pd.DataFrame,
    px_main: pd.DataFrame,
    months_list=(1, 3, 6),
    px_spy: pd.DataFrame | None = None,
    px_qqq: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols = ["date", "title", "url"] + [f"ret_{m}m" for m in months_list]
    if px_spy is not None:
        cols += [f"spy_{m}m" for m in months_list]
    if px_qqq is not None:
        cols += [f"qqq_{m}m" for m in months_list]

    if first_cuts.empty or px_main.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in first_cuts.iterrows():
        d = r.date
        entry = {"date": d, "title": r.get("title", ""), "url": r.get("url", "")}
        for m in months_list:
            entry[f"ret_{m}m"] = fwd_return(px_main, d, m)
            if px_spy is not None:
                entry[f"spy_{m}m"] = fwd_return(px_spy, d, m)
            if px_qqq is not None:
                entry[f"qqq_{m}m"] = fwd_return(px_qqq, d, m)
        rows.append(entry)

    df = pd.DataFrame(rows).sort_values("date")
    return df


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Controls")
ticker = st.sidebar.text_input(
    "Primary Index/ETF (use ^GSPC, SPY, or QQQ)",
    "^GSPC",
)
show_spy = st.sidebar.checkbox("Add SPY", True)
show_qqq = st.sidebar.checkbox("Add QQQ", True)
cooldown = st.sidebar.slider(
    "Months between cuts",
    1,
    12,
    FIRST_CUT_COOLDOWN_MONTHS,
)
months_list = st.sidebar.multiselect(
    "Return horizons",
    [1, 3, 6, 9, 12],
    [1, 3, 6],
)
start_year = st.sidebar.number_input(
    "Min year",
    1980,
    datetime.now().year,
    1990,
)
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# -------------------------------
# Data pipeline
# -------------------------------
st.title("📉 Fed Rate Cut Tracker — Market Dashboard")

moves = fetch_fed_press_releases()
moves["date"] = pd.to_datetime(
    moves.get("date", pd.Series(dtype="datetime64[ns]")),
    errors="coerce",
)
moves = moves.dropna(subset=["date"])
if not moves.empty:
    moves = moves[moves.date.dt.year >= start_year].reset_index(drop=True)
else:
    st.warning(
        "No policy statements parsed from feed "
        "(check network or RSS changes)."
    )

first_cuts = identify_first_cuts(moves, cooldown)
px_df = fetch_prices(ticker, start=f"{start_year}-01-01")

# policy rates and annotations
rate_df = fetch_policy_rates(start=f"{start_year}-01-01")
first_cuts_rates = annotate_rates(first_cuts, rate_df)

# optional SPY/QQQ series for table
spy_df = (
    fetch_prices("SPY", start=f"{start_year}-01-01")
    if show_spy else
    pd.DataFrame()
)
qqq_df = (
    fetch_prices("QQQ", start=f"{start_year}-01-01")
    if show_qqq else
    pd.DataFrame()
)

returns_df = compute_returns(
    first_cuts_rates,
    px_df,
    months_list,
    px_spy=spy_df if not spy_df.empty else None,
    px_qqq=qqq_df if not qqq_df.empty else None,
)

# -------------------------------
# Tables
# -------------------------------
st.subheader("Forward returns per first cut")
if returns_df.empty:
    st.info(
        "No first cuts detected in the selected period. Try widening the "
        "date range or lowering the cooldown."
    )
else:
    show_cols = [
        "date", "title", "rate_lower", "rate_upper",
        "rate_mid", "eff_rate", "delta_mid_bps",
    ]
    df_show = first_cuts_rates[show_cols].merge(
        returns_df,
        on="date",
        how="left",
        suffixes=("", ""),
    )
    df_show = df_show.sort_values("date")
    df_show["date"] = df_show["date"].dt.date
    st.dataframe(df_show)

    sel = [m for m in [1, 3, 6] if f"ret_{m}m" in returns_df.columns]
    if sel:
        cols = ["date", "title"] + [f"ret_{m}m" for m in sel]
        sub = returns_df[cols].copy()
        sub.columns = ["date", "title"] + [f"{m}m_%" for m in sel]
        st.markdown("**1/3/6-month summary (primary)**")
        st.dataframe(sub.assign(date=sub.date.dt.date))

# -------------------------------
# SPY/QQQ cycle performance
# -------------------------------
if not first_cuts.empty:
    cycles, bars = [], []
    for i, r in first_cuts.reset_index(drop=True).iterrows():
        start_dt = r.date
        if i + 1 < len(first_cuts):
            end_dt = first_cuts.iloc[i + 1].date - pd.Timedelta(days=1)
        else:
            end_dt = pd.Timestamp.today()
        cycles.append(
            {
                "start": start_dt,
                "end": end_dt,
                "label": start_dt.strftime("%Y-%m-%d"),
            }
        )
    for c in cycles:
        row = {"cycle": c["label"]}
        for lbl, dfx in [("SPY", spy_df), ("QQQ", qqq_df)]:
            if dfx.empty:
                row[lbl] = np.nan
                continue
            s = nearest_close(dfx, c["start"])
            e = nearest_close(dfx, c["end"])
            ok = not (np.isnan(s) or np.isnan(e))
            row[lbl] = (e / s - 1) * 100 if ok else np.nan
        bars.append(row)
    perf = pd.DataFrame(bars)
    if not perf.empty:
        st.subheader("Cycle returns (SPY/QQQ)")
        st.dataframe(perf)
else:
    st.caption("Cycle performance table hidden — no first cuts in range.")

# -------------------------------
# Recession overlay chart
# -------------------------------
st.subheader("Index with recessions, rates & cut markers")
fig = go.Figure()
if not px_df.empty:
    fig.add_trace(
        go.Scatter(
            x=px_df.index,
            y=px_df["close"],
            name=ticker,
        )
    )

try:
    usrec = fred_csv("USREC", start=f"{start_year}-01-01")
except Exception as e:
    st.caption(f"USREC fetch failed: {e}")
    usrec = pd.DataFrame()

if not usrec.empty:
    in_rec = False
    rec_start = None
    chart_end = (
        px_df.index.max() if not px_df.empty else pd.Timestamp.today()
    )
    for d, v in usrec["USREC"].items():
        if v == 1 and not in_rec:
            in_rec, rec_start = True, d
        elif v == 0 and in_rec:
            fig.add_vrect(
                x0=rec_start,
                x1=d,
                fillcolor="gray",
                opacity=0.15,
            )
            in_rec = False
    if in_rec:
        fig.add_vrect(
            x0=rec_start,
            x1=chart_end,
            fillcolor="gray",
            opacity=0.15,
        )

# add policy rate on a secondary axis
if not rate_df.empty:
    have_band = {"DFEDTARU", "DFEDTARL"}.issubset(rate_df.columns)
    if have_band:
        mid = (rate_df["DFEDTARU"] + rate_df["DFEDTARL"]) / 2.0
        fig.add_trace(
            go.Scatter(
                x=mid.index,
                y=mid.values,
                name="Policy rate (mid, %)",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rate_df.index,
                y=rate_df["DFEDTARU"],
                name="Target upper",
                yaxis="y2",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rate_df.index,
                y=rate_df["DFEDTARL"],
                name="Target lower",
                yaxis="y2",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(150,150,150,0.2)",
                showlegend=False,
            )
        )
    elif "FEDFUNDS" in rate_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rate_df.index,
                y=rate_df["FEDFUNDS"],
                name="Effective Fed Funds (%)",
                yaxis="y2",
            )
        )

for _, r in first_cuts.iterrows():
    fig.add_vline(
        x=r.date,
        line=dict(color="red", dash="dash"),
    )

fig.update_layout(
    yaxis=dict(title=ticker),
    yaxis2=dict(
        title="Policy rate (%)",
        overlaying="y",
        side="right",
    ),
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Macro fallback (simple chart)
# -------------------------------
st.subheader("Macro: CPI, Core CPI, Unemployment (FRED)")
try:
    fred = fred_csv(
        ["CPIAUCSL", "CPILFESL", "UNRATE"],
        start=f"{start_year}-01-01",
    )
    fred = fred.rename(
        columns={
            "CPIAUCSL": "CPI",
            "CPILFESL": "Core CPI",
            "UNRATE": "Unemployment",
        }
    )
    st.line_chart(fred)
except Exception as e:
    st.warning(f"Macro fetch failed: {e}")

# -------------------------------
# Sanity tests
# -------------------------------
with st.expander("🔧 Run sanity tests"):
    if st.button("Run tests now"):
        try:
            cols_ok = {"date", "action", "title", "url"}
            assert cols_ok.issubset(moves.columns), "moves columns missing"
            is_dt = pd.api.types.is_datetime64_any_dtype(moves["date"])
            assert is_dt or moves.empty, "moves.date not datetime"
            assert isinstance(first_cuts, pd.DataFrame), \
                "first_cuts not DataFrame"
            if not first_cuts.empty:
                ok_sorted = (
                    first_cuts["date"].diff().dropna() >= pd.Timedelta(0)
                ).all()
                assert ok_sorted, "first_cuts not sorted"
            assert isinstance(returns_df, pd.DataFrame), \
                "returns_df not DataFrame"
            st.success("All sanity tests passed.")
        except AssertionError as ae:
            st.error(f"Test failed: {ae}")
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")
