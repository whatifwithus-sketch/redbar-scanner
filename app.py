import datetime as dt
import json
import os

import streamlit as st
import streamlit.components.v1 as components

import redbar_scanner as rs

st.set_page_config(page_title="Red Bar Scanner", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding:0.6rem 0.6rem}</style>", unsafe_allow_html=True)

demo = st.sidebar.checkbox("Demo data (fake, for testing)", value=False)


def universe():
    # Optional: add a file called universe.txt (one symbol per line) to use your own list
    if os.path.exists("universe.txt"):
        return rs.load_universe("universe.txt")
    return rs.DEFAULT


@st.cache_data(ttl=300, show_spinner="Scanning NSE stocks... (about a minute)")
def run_scan(use_demo):
    if use_demo:
        data, bad = rs.demo_data(), []
    else:
        data, bad = rs.fetch(universe() + ["^NSEI"])
    nifty = rs.analyze("NIFTY", data["^NSEI"]) if "^NSEI" in data else None
    rows = [r for s, df in data.items() if s != "^NSEI" for r in [rs.analyze(s, df)] if r]
    meta = rs.market_regime(rows, nifty) if rows else None
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return rows, meta, bad, dt.datetime.now(ist).strftime("%d %b %H:%M IST")


rows, meta, bad, stamp = run_scan(demo)

c1, c2 = st.columns([3, 1])
c1.markdown(f"**Red Bar Scanner** - data fetched {stamp}")
if c2.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

if not rows:
    st.error("No data received. Yahoo may be busy or blocking this server. Tap Refresh in a few minutes.")
    st.stop()

ranked = [r for r in rows if r["status"] in ("ABOVE", "WITHIN", "BELOW")]
html = rs.HTML.replace("__DATA__", json.dumps(rs.jsonable(ranked))).replace("__META__", json.dumps(meta))
components.html(html, height=1700, scrolling=True)
if bad:
    st.caption("No data for: " + ", ".join(bad[:40]))
