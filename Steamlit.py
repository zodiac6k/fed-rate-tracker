# streamlit_app.py — Fed Rate Cut Tracker & Macro Dashboard (no yfinance)
# ---------------------------------------------------------------------------------
# ✅ This build removes the yfinance dependency.
# Data sources:
#   - Fed press releases (Atom feed) + HTML fallback
#   - Index/ETF prices: FRED CSV for S&P 500 (SP500), Stooq CSV for SPY/QQQ
#   - US recessions & macro: FRED CSV (no API key)
#
# Run locally:
#   pip install streamlit pandas numpy requests beautifulsoup4 plotly python-dateutil
#   streamlit run streamlit_app.py

import re, io
import numpy as np, pandas as pd, requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import streamlit as st
import plotly.express as px, plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Config
st.set_page_config(page_title="Fed Rate Cut Tracker", page_icon="📉", layout="wide")

# Feeds & CSV endpoints
RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"  # ✅ correct feed
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
STQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

# Classification patterns for statements
CUT_PATTERNS  = [r"decided to lower the target range", r"lowered the federal funds rate", r"reduce the target range"]
HIKE_PATTERNS = [r"decided to raise the target range",  r"raised the federal funds rate",  r"increase the target range"]
HOLD_PATTERNS = [r"decided to maintain the target range", r"kept the target range", r"maintain its target range"]
FIRST_CUT_COOLDOWN_MONTHS = 4

# -------------------------------
# Helpers — FRED & Stooq (no API keys)
# -------------------------------

def fred_csv(series_ids, start="1980-01-01") -> pd.DataFrame:
    """Fetch one or more FRED series via fredgraph CSV. Returns wide DataFrame indexed by date."""
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    params = {"id": ",".join(series_ids)}
    r = requests.get(FRED_CSV_BASE, params=params, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])  # drop rows with bad dates
    df = df[df["DATE"] >= pd.to_datetime(start)]
    df.set_index("DATE", inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def fetch_stooq_daily(symbol: str, start: str = "1985-01-01") -> pd.DataFrame:
    """Download daily OHLCV from Stooq for symbols like 'spy.us', 'qqq.us'."""
    url = STQ_DAILY_URL.format(symbol=symbol)
    r = requests.get(url, timeout=20)
    if r.status_code != 200 or not r.text or r.text.lower().startswith("html"):
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[df["Date"] >= pd.to_datetime(start)]
    df = df.rename(columns={"Date":"date", "Close":"close"})
    df.set_index("date", inplace=True)
    return df[["close"]]

@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, start: str = "1985-01-01") -> pd.DataFrame:
    """Unified price fetcher without yfinance.
    - ^GSPC → FRED 'SP500'
    - SPY/QQQ (and other US ETFs) → Stooq 'spy.us' / 'qqq.us'
    """
    t = (ticker or "").strip()
    t_upper = t.upper()

    # S&P 500 index via FRED
    if t_upper in {"^GSPC", "SPX", "^SPX", "S&P500", "SP500"}:
        try:
            df = fred_csv("SP500", start=start)
            if "SP500" in df.columns:
                out = df[["SP500"]].rename(columns={"SP500":"close"})
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
        # Fallback: assume US listing, build like 'ticker.us'
        sym = f"{t_lower}.us" if (t_lower := t.lower()) else None
    if sym:
        return fetch_stooq_daily(sym, start=start)

    return pd.DataFrame()

# -------------------------------
# Fed press releases — Atom feed with HTML fallback
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_fed_press_releases() -> pd.DataFrame:
    rows = []
    # Try Atom feed first
    try:
        r = requests.get(RSS_URL, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for e in root.findall(".//atom:entry", ns):
            title = e.findtext("atom:title", namespaces=ns) or ""
            link_el = e.find("atom:link[@rel='alternate']", ns) or e.find("atom:link", ns)
            link = link_el.attrib.get("href") if link_el is not None else None
            published = e.findtext("atom:published", namespaces=ns) or e.findtext("atom:updated", namespaces=ns) or ""
            published_ts = pd.to_datetime(published, errors="coerce")
            if not link:
                continue
            try:
                html = requests.get(link, timeout=15).text
                text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
            except Exception:
                text = title.lower()
            def _m(pats): return any(re.search(p, text) for p in pats)
            if _m(CUT_PATTERNS): action = "cut"
            elif _m(HIKE_PATTERNS): action = "hike"
            elif _m(HOLD_PATTERNS): action = "hold"
            else: action = "unknown"
            rows.append({"date": published_ts, "action": action, "title": title, "url": link})
    except Exception as e:
        st.warning(f"RSS fetch failed ({e}); attempting HTML fallback…")
        try:
            pr = requests.get("https://www.federalreserve.gov/newsevents/pressreleases.htm", timeout=15).text
            soup = BeautifulSoup(pr, "html.parser")
            for a in soup.select("a"):
                at = (a.get_text(strip=True) or "").lower()
                href = a.get("href") or ""
                if "fomc statement" in at and href.startswith("/newsevents/pressreleases/"):
                    link = f"https://www.federalreserve.gov{href}"
                    page = requests.get(link, timeout=15).text
                    psoup = BeautifulSoup(page, "html.parser")
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", href)
                    dt = pd.NaT
                    if m:
                        y, mo, d = map(int, m.groups())
                        dt = pd.Timestamp(y, mo, d)
                    text = psoup.get_text(" ", strip=True).lower()
                    def _m(pats): return any(re.search(p, text) for p in pats)
                    if _m(CUT_PATTERNS): action = "cut"
                    elif _m(HIKE_PATTERNS): action = "hike"
                    elif _m(HOLD_PATTERNS): action = "hold"
                    else: action = "unknown"
                    title = psoup.title.get_text(strip=True) if psoup.title else "FOMC statement"
                    rows.append({"date": dt, "action": action, "title": title, "url": link})
        except Exception as e2:
            st.error(f"Fallback scrape failed: {e2}")

    df = pd.DataFrame(rows, columns=["date","action","title","url"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        mask = df["title"].str.contains("FOMC statement|Federal Open Market Committee", case=False, na=False)
        df = df.loc[mask].reset_index(drop=True)
    return df

# -------------------------------
# Return math
# -------------------------------

def nearest_close(df: pd.DataFrame, dt: pd.Timestamp) -> float:
    if df.empty: return np.nan
    if dt in df.index:
        return float(df.loc[dt, "close"]) if "close" in df.columns else float(df.loc[dt, "adj close"]) 
    idx = df.index.searchsorted(dt)
    if idx < len(df.index):
        row = df.iloc[idx]
        return float(row.get("close", row.get("adj close", np.nan)))
    return np.nan

def fwd_return(df: pd.DataFrame, start: pd.Timestamp, months: int) -> float:
    s = nearest_close(df, start)
    e = nearest_close(df, start + relativedelta(months=months))
    if np.isnan(s) or np.isnan(e): return np.nan
    return (e/s - 1) * 100

def identify_first_cuts(df: pd.DataFrame, cooldown_months: int = FIRST_CUT_COOLDOWN_MONTHS) -> pd.DataFrame:
    if df.empty or "action" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["date","action","title","url"])  # empty schema
    cuts = df[df.action == "cut"].copy()
    if cuts.empty:
        return pd.DataFrame(columns=["date","action","title","url"])  # no cuts
    cuts["prev"] = cuts["date"].shift(1)
    def _is_first(r):
        if pd.isna(r.prev): return True
        return (r.date - r.prev) >= pd.DateOffset(months=cooldown_months)
    cuts["is_first"] = cuts.apply(_is_first, axis=1)
    return cuts[cuts.is_first].drop(columns=["prev","is_first"], errors="ignore").reset_index(drop=True)

def compute_returns(first_cuts: pd.DataFrame, df: pd.DataFrame, months_list=(1,3,6)) -> pd.DataFrame:
    if first_cuts.empty or df.empty:
        cols = ["date","title","url"] + [f"ret_{m}m" for m in months_list]
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in first_cuts.iterrows():
        entry = {"date": r.date, "title": r.get("title",""), "url": r.get("url","")}
        for m in months_list:
            entry[f"ret_{m}m"] = fwd_return(df, r.date, m)
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("date")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Primary Index/ETF (use ^GSPC, SPY, or QQQ)", "^GSPC")
show_spy = st.sidebar.checkbox("Add SPY", True)
show_qqq = st.sidebar.checkbox("Add QQQ", True)
cooldown = st.sidebar.slider("Months between cuts", 1, 12, FIRST_CUT_COOLDOWN_MONTHS)
months_list = st.sidebar.multiselect("Return horizons", [1,3,6,9,12], [1,3,6])
start_year = st.sidebar.number_input("Min year", 1980, datetime.now().year, 1990)
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# -------------------------------
# Data pipeline
# -------------------------------
st.title("📉 Fed Rate Cut Tracker — Market Dashboard")

moves = fetch_fed_press_releases()
moves["date"] = pd.to_datetime(moves.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
moves = moves.dropna(subset=["date"]) 
if not moves.empty:
    moves = moves[moves.date.dt.year >= start_year].reset_index(drop=True)
else:
    st.warning("No policy statements parsed from feed (check network or RSS changes).")

first_cuts = identify_first_cuts(moves, cooldown)
px_df = fetch_prices(ticker, start=f"{start_year}-01-01")
returns_df = compute_returns(first_cuts, px_df, months_list)

# -------------------------------
# Tables
# -------------------------------
st.subheader("Forward returns per first cut")
if returns_df.empty:
    st.info("No first cuts detected in the selected period. Try widening the date range or lowering the cooldown.")
else:
    st.dataframe(returns_df.assign(date=returns_df.date.dt.date))
    sel = [m for m in [1,3,6] if f"ret_{m}m" in returns_df.columns]
    if sel:
        sub = returns_df[["date","title"]+[f"ret_{m}m" for m in sel]].copy()
        sub.columns = ["date","title"]+[f"{m}m_%" for m in sel]
        st.markdown("**1/3/6-month summary**")
        st.dataframe(sub.assign(date=sub.date.dt.date))

# -------------------------------
# SPY/QQQ cycle performance
# -------------------------------
if not first_cuts.empty:
    spy_df = fetch_prices("SPY", start=f"{start_year}-01-01") if show_spy else pd.DataFrame()
    qqq_df = fetch_prices("QQQ", start=f"{start_year}-01-01") if show_qqq else pd.DataFrame()
    cycles, bars = [], []
    for i, r in first_cuts.reset_index(drop=True).iterrows():
        start = r.date
        end = first_cuts.iloc[i+1].date - pd.Timedelta(days=1) if i+1 < len(first_cuts) else pd.Timestamp.today()
        cycles.append({"start": start, "end": end, "label": start.strftime("%Y-%m-%d")})
    for c in cycles:
        row = {"cycle": c["label"]}
        for lbl, dfx in [("SPY", spy_df), ("QQQ", qqq_df)]:
            if dfx.empty: row[lbl] = np.nan; continue
            s = nearest_close(dfx, c["start"]); e = nearest_close(dfx, c["end"])
            row[lbl] = (e/s - 1) * 100 if not (np.isnan(s) or np.isnan(e)) else np.nan
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
st.subheader("Index with recessions & cut markers")
fig = go.Figure()
if not px_df.empty:
    fig.add_trace(go.Scatter(x=px_df.index, y=px_df["close"], name=ticker))
try:
    usrec = fred_csv("USREC", start=f"{start_year}-01-01")
except Exception as e:
    st.caption(f"USREC fetch failed: {e}")
    usrec = pd.DataFrame()
if not usrec.empty:
    in_rec = False; rec_start = None
    for d, v in usrec["USREC"].items():
        if v == 1 and not in_rec: in_rec, rec_start = True, d
        elif v == 0 and in_rec: fig.add_vrect(x0=rec_start, x1=d, fillcolor="gray", opacity=0.15); in_rec = False
    if in_rec: fig.add_vrect(x0=rec_start, x1=px_df.index.max(), fillcolor="gray", opacity=0.15)
for _, r in first_cuts.iterrows():
    fig.add_vline(x=r.date, line=dict(color="red", dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Macro fallback (simple chart)
# -------------------------------
st.subheader("Macro: CPI, Core CPI, Unemployment (FRED)")
try:
    fred = fred_csv(["CPIAUCSL","CPILFESL","UNRATE"], start=f"{start_year}-01-01")
    fred = fred.rename(columns={"CPIAUCSL":"CPI","CPILFESL":"Core CPI","UNRATE":"Unemployment"})
    st.line_chart(fred)
except Exception as e:
    st.warning(f"Macro fetch failed: {e}")

# -------------------------------
# Sanity tests
# -------------------------------
with st.expander("🔧 Run sanity tests"):
    if st.button("Run tests now"):
        try:
            assert set(["date","action","title","url"]).issubset(moves.columns), "moves columns missing"
            assert pd.api.types.is_datetime64_any_dtype(moves["date"]) or moves.empty, "moves.date not datetime"
            assert isinstance(first_cuts, pd.DataFrame), "first_cuts not DataFrame"
            if not first_cuts.empty:
                assert (first_cuts["date"].diff().dropna() >= pd.Timedelta(0)).all(), "first_cuts not sorted"
            assert isinstance(returns_df, pd.DataFrame), "returns_df not DataFrame"
            st.success("All sanity tests passed.")
        except AssertionError as ae:
            st.error(f"Test failed: {ae}")
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")
