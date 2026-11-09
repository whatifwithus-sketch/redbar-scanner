#!/usr/bin/env python3
"""
redbar_scanner.py - NSE "Red Bar" scanner (approximation of the MAHAKAAL dashboard)

Red Bar rule used (read from the scanner screenshot: "RB #1 09:20", "RB #4 09:35"):
  * 5-minute candles, the 09:15 candle is IGNORED
  * Red Bar = FIRST RED 5m candle from 09:20 onwards; RB # = its position (09:20 = #1)
  * Zone = that candle's High/Low, with 44/50/56% levels
  * Last price ABOVE the zone = bullish, BELOW = bearish, inside = WITHIN (watch)
  * Targets Tn = High + n x range (up) or Low - n x range (down)
Ranking/score is MY design (not the original's) - see score_row().

Setup:   pip install pandas numpy yfinance
Run:     python redbar_scanner.py                 (built-in list)
         python redbar_scanner.py --universe nse  (official NSE F&O list, needs internet)
         python redbar_scanner.py --universe my.txt   (one symbol per line)
         python redbar_scanner.py --day 2026-11-04    (past day)
         python redbar_scanner.py --demo              (fake data, test the pipeline)
Output:  redbar_scan.csv and redbar_dashboard.html (open in your browser)
"""
import argparse
import datetime as dt
import io
import json
import os

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
TF_MIN = 5
OPEN_T, LAST_T = dt.time(9, 15), dt.time(15, 29)

DEFAULT = ("RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT KOTAKBANK AXISBANK HINDUNILVR "
           "BAJFINANCE MARUTI SUNPHARMA NTPC POWERGRID ONGC TITAN ULTRACEMCO ASIANPAINT WIPRO HCLTECH M&M "
           "ADANIENT ADANIPORTS COALINDIA JSWSTEEL TATASTEEL TECHM NESTLEIND BAJAJFINSV HINDALCO GRASIM "
           "CIPLA DRREDDY EICHERMOT BRITANNIA APOLLOHOSP DIVISLAB BPCL INDUSINDBK HEROMOTOCO TATACONSUM "
           "SBILIFE HDFCLIFE BAJAJ-AUTO SHRIRAMFIN TRENT BEL HAL VEDL SRF MARICO ALKEM BIOCON BSE SUZLON "
           "VBL AMBER DLF PNB BANKBARODA CANBK IOC GAIL TATAPOWER JINDALSTEL SAIL NMDC PFC RECLTD IRCTC "
           "DABUR GODREJCP PIDILITIND HAVELLS SIEMENS ABB LUPIN AUROPHARMA TORNTPHARM ZYDUSLIFE MAXHEALTH "
           "PERSISTENT COFORGE MPHASIS DIXON POLYCAB CUMMINSIND BHARATFORG ASHOKLEY TVSMOTOR MOTHERSON "
           "BOSCHLTD UPL PIIND COLPAL ICICIGI SBICARD CHOLAFIN MUTHOOTFIN LICHSGFIN IDFCFIRSTB FEDERALBNK "
           "BANDHANBNK AUBANK YESBANK IDEA INDIGO JIOFIN OBEROIRLTY GODREJPROP LODHA NHPC ADANIGREEN "
           "ABCAPITAL ASTRAL AMBUJACEM SHREECEM LAURUSLABS GLENMARK").split()


# ----------------------------------------------------------------------------
# Universe + data
# ----------------------------------------------------------------------------
def nse_fno_symbols():
    import requests
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skipinitialspace=True)
    col = [c for c in df.columns if c.strip().upper() == "SYMBOL"][0]
    skip = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SYMBOL", "NAN"}
    return [s.strip() for s in df[col].astype(str) if s.strip().upper() not in skip]


def load_universe(spec):
    if spec == "default":
        return DEFAULT
    if spec == "nse":
        try:
            return nse_fno_symbols()
        except Exception as e:
            print(f"Could not fetch NSE list ({e}); using built-in list.")
            return DEFAULT
    with open(spec) as f:
        return [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]


def yf_sym(s):
    return s if s.startswith("^") else s + ".NS"


def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            vals = [str(v).lower() for v in df.columns.get_level_values(lvl)]
            if "close" in vals:
                df.columns = vals
                break
    df = df.rename(columns=lambda c: str(c).lower())
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    idx = df.index
    df.index = idx.tz_convert(IST) if idx.tz is not None else idx.tz_localize(IST)
    return df.sort_index().between_time(OPEN_T, LAST_T)


def fetch(symbols, period="7d", batch=40):
    import yfinance as yf
    data, bad = {}, []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        tick = [yf_sym(s) for s in chunk]
        try:
            raw = yf.download(tick, interval=f"{TF_MIN}m", period=period, group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print("  batch failed:", e)
            bad += chunk
            continue
        for s, t in zip(chunk, tick):
            try:
                df = clean((raw[t] if len(tick) > 1 else raw).copy())
                if len(df) > 30:
                    data[s] = df
                else:
                    bad.append(s)
            except Exception:
                bad.append(s)
        print(f"  downloaded {min(i + batch, len(symbols))}/{len(symbols)}")
    return data, bad


def demo_data(n=40, days=5, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    dates = pd.bdate_range(end=pd.Timestamp.now(tz=IST).normalize(), periods=days)
    for s in DEFAULT[:n] + ["^NSEI"]:
        p, drift, frames = rng.uniform(100, 5000), rng.normal(0, 0.0004), []
        for d in dates:
            idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq=f"{TF_MIN}min")
            c = p * np.exp(np.cumsum(rng.normal(drift, 0.0018, 75)))
            o = np.r_[c[0] * (1 + rng.normal(0, 0.001)), c[:-1]]
            hi = np.maximum(o, c) * (1 + rng.uniform(0, 0.0012, 75))
            lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.0012, 75))
            frames.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                                        "volume": rng.integers(2000, 60000, 75)}, index=idx))
            p = c[-1]
        out[s] = pd.concat(frames)
    return out


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def supertrend_dir(df, n=14, mult=3.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb, d = ub.copy(), lb.copy(), 1
    for i in range(1, len(c)):
        if np.isnan(atr[i]):
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1] or np.isnan(fub[i - 1])) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1] or np.isnan(flb[i - 1])) else flb[i - 1]
        if d == 1 and c[i] < flb[i]:
            d = -1
        elif d == -1 and c[i] > fub[i]:
            d = 1
    return d


def score_row(direction, last, H, L, R, st, vwap, chg, relvol, reached, rr):
    """0-100 score for a long (+1) or short (-1) setup. MY weights, edit freely."""
    s = 30                                                        # outside the Red Bar
    s += 10 if st == direction else 0                             # Supertrend agrees
    s += 10 if (not np.isnan(vwap) and (last > vwap if direction == 1 else last < vwap)) else 0
    s += 10 if (not np.isnan(chg) and chg * direction > 0) else 0  # day change agrees
    s += 15 if relvol >= 1.5 else 8 if relvol >= 1.0 else 0       # volume vs same time earlier days
    s += 15 if rr >= 2 else 8 if rr >= 1 else 0                   # reward:risk to next target (SL = midpoint)
    s += 10 if reached <= 1 else 5 if reached == 2 else 0         # not already extended
    return s


def analyze(sym, df, day=None, tstep=1.0):
    dates = pd.Series(df.index.date, index=df.index)
    avail = sorted(set(dates))
    day = day or avail[-1]
    if day not in avail:
        return None
    keep = dates <= day
    df, dates = df[keep], dates[keep]
    today, prev = df[dates == day], df[dates < day]
    if len(today) < 3:
        return None
    last = float(today["close"].iloc[-1])
    prev_close = float(prev["close"].iloc[-1]) if len(prev) else np.nan
    chg = (last / prev_close - 1) * 100 if prev_close == prev_close else np.nan
    row = dict(symbol=sym, day=str(day), price=round(last, 2), chg_pct=round(chg, 2) if chg == chg else None,
               last_time=today.index[-1].strftime("%H:%M"), status="NO RED BAR", side="-", score=0, grade="")

    cand = today.iloc[1:]                                # ignore the first (09:15) candle
    reds = cand[cand["close"] < cand["open"]]
    if reds.empty:
        return row
    rb_t, rb = reds.index[0], reds.iloc[0]
    H, L = float(rb["high"]), float(rb["low"])
    R = H - L
    if R <= 0:
        return row
    after = today[today.index > rb_t]
    hi = float(after["high"].max()) if len(after) else H
    lo = float(after["low"].min()) if len(after) else L
    up_r = int(max(0, np.floor((hi - H) / (tstep * R) + 1e-9)))
    dn_r = int(max(0, np.floor((L - lo) / (tstep * R) + 1e-9)))
    status = "ABOVE" if last > H else "BELOW" if last < L else "WITHIN"
    direction = 1 if status == "ABOVE" else -1 if status == "BELOW" else 0

    st = supertrend_dir(df)
    vol = today["volume"].cumsum()
    tp = (today["high"] + today["low"] + today["close"]) / 3
    vwap = float((tp * today["volume"]).cumsum().iloc[-1] / vol.iloc[-1]) if vol.iloc[-1] > 0 else np.nan
    tod = today.index[-1].time()
    pv = [g[g.index.time <= tod]["volume"].sum() for _, g in prev.groupby(prev.index.date)]
    relvol = float(vol.iloc[-1] / np.mean(pv)) if pv and np.mean(pv) > 0 else np.nan

    row.update(status=status,
               rb_no=int((rb_t - today.index[0]).total_seconds() // 60 // TF_MIN),
               rb_time=rb_t.strftime("%H:%M"), rb_high=round(H, 2), rb_low=round(L, 2),
               l44=round(L + .44 * R, 2), l50=round(L + .50 * R, 2), l56=round(L + .56 * R, 2),
               pos_pct=round((last - L) / R * 100, 1), supertrend="UP" if st == 1 else "DOWN",
               vs_vwap=("ABOVE" if last > vwap else "BELOW") if vwap == vwap else "-",
               relvol=round(relvol, 2) if relvol == relvol else None)
    if direction:
        reached = up_r if direction == 1 else dn_r
        nxt = H + (reached + 1) * tstep * R if direction == 1 else L - (reached + 1) * tstep * R
        risk = abs(last - (H + L) / 2)
        rr = abs(nxt - last) / risk if risk > 0 else np.nan
        sc = score_row(direction, last, H, L, R, st, vwap, chg, relvol if relvol == relvol else 0,
                       reached, rr if rr == rr else 0)
        row.update(side="LONG" if direction == 1 else "SHORT", targets_hit=reached,
                   next_target=round(nxt, 2), sl_mid=round((H + L) / 2, 2),
                   rr_next=round(rr, 2) if rr == rr else None, score=sc,
                   grade="A" if sc >= 75 else "B" if sc >= 60 else "C")
    else:
        row["side"] = "WATCH"
    return row


def market_regime(rows, nifty):
    v = [r for r in rows if r["status"] in ("ABOVE", "WITHIN", "BELOW")]
    n = max(len(v), 1)
    a = sum(r["status"] == "ABOVE" for r in v)
    w = sum(r["status"] == "WITHIN" for r in v)
    b = sum(r["status"] == "BELOW" for r in v)
    nf = nifty["status"] if nifty else "n/a"
    if a / n >= .5 and nf != "BELOW":
        reg, tip = "BULLISH", "Favour LONG setups (ABOVE the Red Bar)."
    elif b / n >= .5 and nf != "ABOVE":
        reg, tip = "BEARISH", "Favour SHORT setups (BELOW the Red Bar)."
    else:
        reg, tip = "MIXED", "No clear side. Trade only grade A, or wait."
    return dict(above=a, within=w, below=b, total=len(v), nifty=nf, regime=reg, tip=tip)


# ----------------------------------------------------------------------------
# HTML dashboard
# ----------------------------------------------------------------------------
HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Red Bar Scanner</title><style>
:root{--bg:#0b0f10;--card:#12181a;--t:#e8eceb;--m:#8a9794;--g:#2ecc71;--r:#ef5350;--y:#f5b942}
*{box-sizing:border-box}body{margin:0;padding:14px;background:var(--bg);color:var(--t);font:14px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.m{color:var(--m)}.bar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
input,select{background:var(--card);color:var(--t);border:1px solid #2a3436;border-radius:8px;padding:8px}
.reg{padding:10px 12px;border-radius:10px;background:var(--card);border-left:4px solid var(--y);margin:8px 0}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.col h2{font-size:15px;margin:6px 0}.card{background:var(--card);border-radius:10px;padding:10px;margin-bottom:8px;border:1px solid #1d2628}
.top{display:flex;justify-content:space-between;align-items:center}.up{color:var(--g)}.dn{color:var(--r)}
.tag{font-size:11px;padding:2px 6px;border-radius:6px;background:#1d2628;color:var(--m)}
.rbar{position:relative;height:6px;background:#1d2628;border-radius:3px;margin:8px 0 4px}
.rbar b{position:absolute;left:33.3%;width:33.4%;top:0;bottom:0;background:#7a2a2a;border-radius:3px}
.rbar i{position:absolute;top:-3px;width:3px;height:12px;background:#fff;border-radius:2px}
.sec{font-size:12px;color:var(--m)}
</style></head><body>
<h1>Red Bar Scanner <span class="m" id="stamp"></span></h1>
<div class="reg" id="reg"></div>
<div class="bar"><input id="q" placeholder="Search symbol..."><select id="s"><option value="score">Sort: score</option>
<option value="chg">Sort: % change</option><option value="sym">Sort: A-Z</option></select>
<select id="g"><option value="">All grades</option><option value="A">Grade A</option><option value="B">Grade B+</option></select></div>
<div class="cols" id="cols"></div>
<p class="m">Approximation built from screenshots. Not investment advice. Test on paper first.</p>
<script>
const D=__DATA__, M=__META__;
document.getElementById('stamp').textContent=' - '+D[0].day+' (last '+D[0].last_time+')';
document.getElementById('reg').innerHTML='<b>Market: '+M.regime+'</b> &nbsp; Above '+M.above+' / Within '+M.within+' / Below '+M.below+' &nbsp; Nifty: '+M.nifty+'<br><span class="m">'+M.tip+'</span>';
const cols=[['ABOVE','Above Red Bar (bullish)'],['WITHIN','Within Red Bar (wait)'],['BELOW','Below Red Bar (bearish)']];
function card(r){const p=Math.max(0,Math.min(100,(r.pos_pct+100)/3));const c=r.chg_pct;
return '<div class="card"><div class="top"><b>'+r.symbol+'</b><span class="tag">'+(r.grade?('Grade '+r.grade+' · '+r.score):r.side)+'</span></div>'
+'<div>&#8377;'+r.price+' <span class="'+(c>=0?'up':'dn')+'">'+(c==null?'':(c>0?'+':'')+c+'%')+'</span></div>'
+'<div class="rbar"><b></b><i style="left:'+p+'%"></i></div>'
+'<div class="sec">RB #'+r.rb_no+' '+r.rb_time+' · H '+r.rb_high+' / L '+r.rb_low+'</div>'
+(r.next_target?'<div class="sec">Next T '+r.next_target+' · SL(mid) '+r.sl_mid+' · R:R '+r.rr_next+' · hit '+r.targets_hit+'</div>':'')
+'<div class="sec">ST '+r.supertrend+' · VWAP '+r.vs_vwap+' · RelVol '+(r.relvol==null?'-':r.relvol)+'</div></div>';}
function draw(){const q=document.getElementById('q').value.toUpperCase(),s=document.getElementById('s').value,g=document.getElementById('g').value;
let h='';for(const [k,t] of cols){let L=D.filter(r=>r.status==k&&r.symbol.includes(q)&&(!g||(g=='A'?r.grade=='A':(r.grade=='A'||r.grade=='B'))));
L.sort((a,b)=>s=='sym'?a.symbol.localeCompare(b.symbol):s=='chg'?(k=='BELOW'?a.chg_pct-b.chg_pct:b.chg_pct-a.chg_pct):b.score-a.score);
h+='<div class="col"><h2>'+t+' <span class="tag">'+L.length+'</span></h2>'+L.map(card).join('')+'</div>';}
document.getElementById('cols').innerHTML=h;}
['q','s','g'].forEach(i=>document.getElementById(i).addEventListener('input',draw));draw();
</script></body></html>"""


def jsonable(rows):
    def fix(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if v != v else float(v)
        return v
    return [{k: fix(v) for k, v in r.items()} for r in rows]


def write_outputs(rows, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "redbar_scan.csv"), index=False)
    ranked = [r for r in rows if r["status"] in ("ABOVE", "WITHIN", "BELOW")]
    html = HTML.replace("__DATA__", json.dumps(jsonable(ranked))).replace("__META__", json.dumps(meta))
    path = os.path.join(out_dir, "redbar_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NSE Red Bar scanner")
    ap.add_argument("--universe", default="default", help="default | nse | path/to/symbols.txt")
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (default: latest day in data)")
    ap.add_argument("--limit", type=int, default=0, help="only scan first N symbols")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=".")
    ap.add_argument("--demo", action="store_true", help="use fake data to test the pipeline")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.day) if a.day else None

    if a.demo:
        data, bad = demo_data(), []
    else:
        syms = load_universe(a.universe)
        if a.limit:
            syms = syms[:a.limit]
        print(f"Downloading {TF_MIN}m data for {len(syms)} symbols (+ Nifty)...")
        data, bad = fetch(syms + ["^NSEI"])
    nifty = analyze("NIFTY", data["^NSEI"], day) if "^NSEI" in data else None
    rows = [r for s, df in data.items() if s != "^NSEI" for r in [analyze(s, df, day)] if r]
    if not rows:
        print("No data. Check internet / symbols / market day.")
        return
    meta = market_regime(rows, nifty)

    print(f"\nDay {rows[0]['day']}  last bar {rows[0]['last_time']} IST   scanned {len(rows)}   failed {len(bad)}")
    print(f"Breadth: ABOVE {meta['above']} | WITHIN {meta['within']} | BELOW {meta['below']}   Nifty: {meta['nifty']}")
    print(f"MARKET: {meta['regime']}  ->  {meta['tip']}")
    cols = ["symbol", "price", "chg_pct", "rb_no", "rb_time", "targets_hit", "next_target", "rr_next", "relvol", "score", "grade"]
    for side, title in (("LONG", "BEST LONGS (above Red Bar)"), ("SHORT", "BEST SHORTS (below Red Bar)")):
        df = pd.DataFrame([r for r in rows if r["side"] == side])
        print(f"\n{title}")
        print("none" if df.empty else df.sort_values("score", ascending=False).head(a.top)[cols].to_string(index=False))
    print("\nSaved:", write_outputs(rows, meta, a.out), "and redbar_scan.csv")
    if bad:
        print("No data for:", ", ".join(bad[:30]), "..." if len(bad) > 30 else "")


if __name__ == "__main__":
    main()

