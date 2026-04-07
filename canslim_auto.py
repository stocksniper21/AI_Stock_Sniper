#!/usr/bin/env python3
import asyncio, json, os, sys, time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright
import anthropic
import google.genai as genai

SCREENER_URL = "http://www.canslimscreener.com/"
TARGET       = 50
TOP_N        = 10
CLAUDE_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
OUTPUT       = Path("canslim_dashboard.html")


async def scrape():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        print(f"Opening {SCREENER_URL} ...")
        await page.goto(SCREENER_URL, wait_until="networkidle", timeout=90_000)
        await page.wait_for_timeout(4_000)
        print("Setting page size to 50...")
        try:
            await page.wait_for_selector("#pageDropDown", timeout=10_000)
            await page.click("#pageDropDown")
            await page.wait_for_timeout(1_000)
            await page.click('ul.dropdown-menu a:has-text("50")')
            await page.wait_for_timeout(4_000)
            print("Page size set to 50")
        except Exception as e:
            print(f"Could not set page size: {e}")
        stocks = await read_table(page)
        await browser.close()
    return stocks


async def read_table(page):
    stocks = []
    rows = await page.query_selector_all("tr")
    print(f"Found {len(rows)} rows")
    headers = []
    for row in rows:
        cells = await row.query_selector_all("th, td")
        values = []
        for c in cells:
            text = (await c.inner_text()).strip().split("\n")[0].strip()
            values.append(text)
        if not values:
            continue
        if not headers and len(values) > 5 and values[0] == "Ticker":
            headers = values
            print(f"Headers found: {headers[:6]}")
            continue
        if headers and len(values) >= 31 and values[0] and values[0] != "Ticker":
            try:
                s = {
                    "ticker":       values[0].upper().strip(),
                    "nval":         int(float(values[1])),
                    "lwowski":      int(float(values[2])),
                    "primaryTests": int(float(values[3])),
                    "sma50_150":    values[5].lower() == "true",
                    "sma150_200":   values[6].lower() == "true",
                    "span52":       values[7].lower() == "true",
                    "rs_rule":      values[8].lower() == "true",
                    "liquidity":    values[9].lower() == "true",
                    "above52h":     values[10].lower() == "true",
                    "prevclose":    values[11].lower() == "true",
                    "sma200slope":  values[12].lower() == "true",
                    "inst_rule":    values[13].lower() == "true",
                    "abovesma50":   values[14].lower() == "true",
                    "sales_rule":   values[15].lower() == "true",
                    "eps_rule":     values[16].lower() == "true",
                    "high52":       round(float(values[17]), 2),
                    "instOwn":      round(float(values[18]) * 100, 1),
                    "salesPct":     round(float(values[19]) * 100, 1),
                    "epsPct":       round(float(values[25]) * 100, 1),
                    "volume":       int(float(values[26])),
                    "low52":        round(float(values[27]), 2),
                    "price":        round(float(values[29]), 2),
                    "rsVal":        int(float(values[30])),
                    "sma200pct":    round(float(values[21]) * 100, 1),
                }
                if s["ticker"] and len(s["ticker"]) <= 7:
                    stocks.append(s)
            except (ValueError, IndexError):
                continue
    print(f"Parsed {len(stocks)} stocks")
    return stocks[:TARGET]


def build_prompt(stocks):
    lines = []
    for i, s in enumerate(stocks, 1):
        lines.append(
            f"{i}. {s['ticker']} | N:{s['nval']} | Tests:{s['primaryTests']}/11 | RS:{s['rsVal']} | "
            f"EPS:{s['epsPct']:+.1f}% | Sales:{s['salesPct']:+.1f}% | Inst:{s['instOwn']}% | "
            f"Price:${s['price']:.2f} | 52wH:${s['high52']:.2f} | 52wL:${s['low52']:.2f} | "
            f"Vol:{s['volume']/1e6:.1f}M | vs200SMA:{s['sma200pct']:+.1f}%"
        )
    body = "\n".join(lines)
    return f"""You are an expert CANSLIM analyst screening for multi-bagger opportunities.

Top {len(stocks)} stocks from CANSLIM screener sorted by N-Value (max 8190):
{body}

Select stocks for a 1-2 year hold. Focus on EPS acceleration, RS under-discovered (<40),
institutional build-up, AI/infrastructure exposure, small/mid cap with room to run.

STRICT FILTER: only include stocks where conviction >= 7.
If fewer than 10 qualify return only those. Do NOT pad with lower conviction stocks.

Reply with ONLY a valid JSON array, no markdown fences:
[{{
  "ticker": "XXXX",
  "rank": 1,
  "verdict": "STRONG BUY",
  "conviction": 9,
  "theme": "short theme label",
  "bull_case": "2 sentence summary.",
  "bull_detail": "Full fundamental analysis: revenue growth, EPS acceleration, margins, institutional accumulation, TAM, multi-bagger potential. Minimum 5 sentences with numbers.",
  "technical_analysis": "Full technical analysis: price vs 50/150/200 SMAs, 52-week position, RS, volume, chart setup. Minimum 4 sentences.",
  "key_risk": "1 sentence summary of biggest risk.",
  "risk_detail": "Full risk analysis: bear case, what could go wrong, metrics to watch, sector/valuation risks. Minimum 4 sentences.",
  "price_target_pct": 85,
  "catalyst": "next specific catalyst.",
  "catalyst_detail": "Catalyst timeline, expected impact, what to watch for. Minimum 3 sentences.",
  "price_target_low": 40,
  "price_target_high": 120
}}]
verdict: STRONG BUY / BUY / SPECULATIVE BUY | conviction: 1-10 | price_target_pct: % upside 1-2 years"""


def parse_picks(raw, stocks):
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        picks = json.loads(raw)
    except Exception as e:
        print(f"  JSON error: {e}")
        picks = []
    ticker_map = {s["ticker"]: s for s in stocks}
    empty = {
        "ticker":"","nval":0,"lwowski":0,"primaryTests":0,"rsVal":0,
        "epsPct":0,"salesPct":0,"instOwn":0,"price":0,"high52":0,"low52":0,
        "volume":0,"sma200pct":0,"sma50_150":False,"sma150_200":False,
        "span52":False,"rs_rule":False,"liquidity":False,"above52h":False,
        "prevclose":False,"sma200slope":False,"inst_rule":False,
        "abovesma50":False,"sales_rule":False,"eps_rule":False
    }
    result = []
    for pick in picks:
        base = ticker_map.get(pick.get("ticker",""), {**empty, "ticker": pick.get("ticker","?")})
        result.append({**base, **pick})
    result = [s for s in result if int(s.get("conviction",0)) >= 7]
    return result


def analyze_claude(stocks):
    if not CLAUDE_KEY:
        print("  ANTHROPIC_API_KEY not set")
        return []
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)
    print(f"  Sending {len(stocks)} stocks to Claude...")
    t0 = time.time()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role":"user","content":build_prompt(stocks)}]
    )
    print(f"  Claude done in {time.time()-t0:.1f}s")
    result = parse_picks(msg.content[0].text, stocks)
    print(f"  Claude picks (>=7): {len(result)}")
    return result


def analyze_gemini(stocks):
    if not GEMINI_KEY:
        print("  GEMINI_API_KEY not set")
        return []
    client_gemini = genai.Client(api_key=GEMINI_KEY)
    
    # send only top 10 to stay within free tier limits
    subset = stocks[:50]
    print(f"  Sending top {len(subset)} stocks to Gemini...")
    t0 = time.time()
    response = client_gemini.models.generate_content(model="gemini-2.5-flash", contents=build_prompt(subset))
    print(f"  Gemini done in {time.time()-t0:.1f}s")
    result = parse_picks(response.text, stocks)
    print(f"  Gemini picks (>=7): {len(result)}")
    return result


def merge_picks(claude_picks, gemini_picks, all_stocks):
    """
    Merge Claude and Gemini picks into one deduplicated list.
    Each entry gets a 'models' field: 'claude', 'gemini', or 'both'.
    Stocks picked by both get highest priority and use Claude's analysis
    with Gemini conviction noted.
    """
    claude_map = {s["ticker"]: s for s in claude_picks}
    gemini_map = {s["ticker"]: s for s in gemini_picks}
    all_tickers_ordered = []
    seen = set()

    # both first (highest conviction)
    for t in claude_map:
        if t in gemini_map and t not in seen:
            all_tickers_ordered.append(t)
            seen.add(t)

    # claude only
    for t in claude_map:
        if t not in seen:
            all_tickers_ordered.append(t)
            seen.add(t)

    # gemini only
    for t in gemini_map:
        if t not in seen:
            all_tickers_ordered.append(t)
            seen.add(t)

    merged = []
    for i, ticker in enumerate(all_tickers_ordered, 1):
        if ticker in claude_map and ticker in gemini_map:
            entry = {**claude_map[ticker], "models": "both", "rank": i}
            entry["gemini_conviction"] = gemini_map[ticker].get("conviction", 0)
        elif ticker in claude_map:
            entry = {**claude_map[ticker], "models": "claude", "rank": i}
        else:
            entry = {**gemini_map[ticker], "models": "gemini", "rank": i}
        merged.append(entry)

    return merged


def build_cards(merged):
    if not merged:
        return "<div style='padding:60px;text-align:center;color:#6b7280;font-size:14px'>No stocks passed the conviction filter.</div>"

    RULES  = ["sma50_150","sma150_200","span52","rs_rule","liquidity","above52h",
              "prevclose","sma200slope","inst_rule","abovesma50","sales_rule","eps_rule"]
    LABELS = ["SMA50>150","SMA150>200","52wkSpan","RS","Liquidity","Near52wH",
              "PrevClose","SMA200up","InstOwn",">SMA50","Sales","EPS"]

    vc = {"STRONG BUY":"#16a34a","BUY":"#2563eb","SPECULATIVE BUY":"#d97706","N/A":"#6b7280"}
    def pc(v): return "#16a34a" if v>0 else "#dc2626" if v<0 else "#6b7280"

    cards = ""
    for s in merged:
        col   = vc.get(s.get("verdict","N/A"),"#6b7280")
        conv  = int(s.get("conviction",0))
        up    = s.get("price_target_pct",0)
        n     = s.get("nval",0)
        eps   = s.get("epsPct",0)
        sales = s.get("salesPct",0)
        inst  = s.get("instOwn",0)
        price = s.get("price",0)
        h52   = s.get("high52",0)
        l52   = s.get("low52",0)
        vol   = s.get("volume",0)
        rs    = s.get("rsVal",0)
        tests = s.get("primaryTests",0)
        lw    = s.get("lwowski",0)
        ph    = round(price/h52*100) if h52 else 0
        np_   = round(n/8190*100)
        cc    = "#16a34a" if conv>=8 else "#2563eb" if conv>=7 else "#d97706"
        models = s.get("models","claude")

        # model badge
        if models == "both":
            gc = s.get("gemini_conviction",0)
            badge = f'<span class="mbadge mboth">&#9733; Claude + Gemini &nbsp;<span style="opacity:.7;font-size:9px">Claude:{conv}/10 · Gemini:{gc}/10</span></span>'
        elif models == "gemini":
            badge = '<span class="mbadge mgemini">&#9670; Gemini pick</span>'
        else:
            badge = '<span class="mbadge mclaude">&#9632; Claude pick</span>'

        pills = ""
        for r, lb in zip(RULES, LABELS):
            cls = "pill-pass" if s.get(r) else "pill-fail"
            pills += f'<span class="{cls}">{lb}</span>'

        verdict  = s.get("verdict","N/A")
        ticker   = s.get("ticker","?")
        rank     = s.get("rank","?")
        theme    = s.get("theme","")
        bull               = s.get("bull_case","")
        bull_detail        = s.get("bull_detail","No additional detail available.")
        technical_analysis = s.get("technical_analysis","No technical analysis available.")
        risk               = s.get("key_risk","")
        risk_detail        = s.get("risk_detail","No additional detail available.")
        catalyst           = s.get("catalyst","")
        catalyst_detail    = s.get("catalyst_detail","No additional detail available.")
        price_target_pct   = s.get("price_target_pct", up)
        price_target_low   = s.get("price_target_low", max(0, up-30))
        price_target_high  = s.get("price_target_high", up+30)

        cards += f"""
<div class="card" data-verdict="{verdict}" data-models="{models}">
  <div class="model-badge-row">{badge}</div>
  <div class="ch">
    <div class="cl">
      <div class="rn">#{rank}</div>
      <div><div class="tn">{ticker}</div><div class="tl">{theme}</div></div>
    </div>
    <div class="cr">
      <span class="vc" style="color:{col};background:{col}18;border:1.5px solid {col}40">{verdict}</span>
      <div class="ul" style="color:{pc(up)}">{up:+}% target</div>
    </div>
  </div>
  <div class="mg">
    <div class="mb"><div class="ml">N-Value</div><div class="mv" style="color:#d97706">{n:,}</div><div class="nb"><div style="width:{np_}%;background:#d97706;height:100%;border-radius:2px"></div></div></div>
    <div class="mb"><div class="ml">Lwowski</div><div class="mv" style="color:#7c3aed">{lw:,}</div></div>
    <div class="mb"><div class="ml">Tests</div><div class="mv" style="color:#16a34a">{tests}/11</div></div>
    <div class="mb"><div class="ml">RS</div><div class="mv" style="color:#2563eb">{rs}</div></div>
    <div class="mb"><div class="ml">EPS%</div><div class="mv" style="color:{pc(eps)}">{eps:+.1f}%</div></div>
    <div class="mb"><div class="ml">Sales%</div><div class="mv" style="color:{pc(sales)}">{sales:+.1f}%</div></div>
    <div class="mb"><div class="ml">Inst Own</div><div class="mv" style="color:#7c3aed">{inst:.1f}%</div></div>
    <div class="mb"><div class="ml">Price</div><div class="mv">${price:.2f}</div></div>
    <div class="mb"><div class="ml">52w High</div><div class="mv">${h52:.2f}</div></div>
    <div class="mb"><div class="ml">52w Low</div><div class="mv">${l52:.2f}</div></div>
    <div class="mb"><div class="ml">vs 52wH</div><div class="mv" style="color:{pc(ph-100)}">{ph}%</div></div>
    <div class="mb"><div class="ml">Volume</div><div class="mv">{vol/1e6:.1f}M</div></div>
  </div>
  <div class="rs"><div class="rt">CANSLIM Rules</div><div class="rp">{pills}</div></div>
   <div class="ag">
     <div class="ab bull"><b class="ai">+</b><div style="flex:1"><div class="al">Bull Case</div><div class="at">{bull}</div><div class="xbtn" onclick="xd(this)">+ full fundamental analysis</div><div class="xbox" style="display:none">{bull_detail}</div></div></div>
     <div class="ab" style="background:#13102a;border-color:#ddd6fe"><b class="ai" style="color:#7c3aed">~</b><div style="flex:1"><div class="al" style="color:#7c3aed">Technical Analysis</div><div class="xbtn" onclick="xd(this)">+ view chart analysis</div><div class="xbox" style="display:none">{technical_analysis}</div></div></div>
     <div class="ab bear"><b class="ai">-</b><div style="flex:1"><div class="al">Key Risk</div><div class="at">{risk}</div><div class="xbtn" onclick="xd(this)">+ full risk analysis</div><div class="xbox" style="display:none">{risk_detail}</div></div></div>
     <div class="ab cat"><b class="ai">*</b><div style="flex:1"><div class="al">Catalyst</div><div class="at">{catalyst}</div><div class="xbtn" onclick="xd(this)">+ catalyst detail</div><div class="xbox" style="display:none">{catalyst_detail}</div></div></div>
   </div>
   <div class="prr"><span class="prl">Price target range:</span><span class="prlo">{price_target_low:+}%</span><span class="prsp">to</span><span class="prhi">{price_target_high:+}%</span><span class="prmi">mid: {price_target_pct:+}%</span></div>
  <div class="convrow">
    <span class="convlbl">Conviction</span>
    <div class="convbar"><div style="width:{conv*10}%;background:{cc};height:100%;border-radius:3px"></div></div>
    <span class="convnum">{conv}/10</span>
  </div>
</div>"""

    return cards


def build_dashboard(merged, total, timestamp):
    if not merged:
        return "<html><body><h1>No picks.</h1></body></html>"

    both_picks  = [s for s in merged if s.get("models") == "both"]
    claude_only = [s for s in merged if s.get("models") == "claude"]
    gemini_only = [s for s in merged if s.get("models") == "gemini"]

    both_chips = " ".join(
        f'<span style="background:#fef3c7;color:#92400e;border:1px solid #fcd34d;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;font-family:monospace">{s["ticker"]}</span>'
        for s in both_picks
    ) if both_picks else '<span style="color:#9ca3af;font-size:13px">No overlap yet — models picked different stocks</span>'

    avg_n   = sum(s.get("nval",0) for s in merged) // len(merged)
    avg_ep  = sum(s.get("epsPct",0) for s in merged) // len(merged)
    avg_up  = sum(s.get("price_target_pct",0) for s in merged) // len(merged)
    cards   = build_cards(merged)

    css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#080d1a;color:#e2e8f0;font-family:'Outfit',sans-serif}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:#1a2540;border-radius:3px}
.header{background:#0c1221;border-bottom:1px solid #1a2540;padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:60px;position:sticky;top:0;z-index:100}
.lw{display:flex;align-items:center;gap:12px}
.lm{width:34px;height:34px;background:#00d4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;color:#080d1a;font-weight:700;font-size:11px;font-family:'JetBrains Mono',monospace}
.lt{font-size:15px;font-weight:700;color:#e2e8f0;letter-spacing:-.02em}.lt span{color:#00d4ff}
.hm{font-size:12px;color:#64748b;text-align:right;line-height:1.8}.hm strong{color:#e2e8f0}
.ld{display:inline-block;width:6px;height:6px;background:#10b981;border-radius:50%;margin-right:5px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.both-box{margin:16px 32px;background:#1a1000;border:1px solid #f59e0b40;border-radius:12px;padding:13px 18px;display:flex;align-items:center;gap:12px}
.both-label{font-size:11px;font-weight:700;color:#f59e0b;text-transform:uppercase;letter-spacing:.08em;white-space:nowrap}
.sum{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:16px 32px}
.sc{background:#0c1221;border:1px solid #1a2540;border-radius:12px;padding:14px 18px;position:relative;overflow:hidden}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.sc:nth-child(1)::before{background:#00d4ff}
.sc:nth-child(2)::before{background:#f59e0b}
.sc:nth-child(3)::before{background:#10b981}
.sc:nth-child(4)::before{background:#8b5cf6}
.sv{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;margin-bottom:3px}
.sl{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.ss{font-size:11px;color:#334155;margin-top:2px}
.toolbar{padding:0 32px 14px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.tbl{font-size:12px;color:#64748b;margin-right:4px}
.fb{padding:5px 14px;border-radius:6px;border:1px solid #1a2540;background:transparent;color:#64748b;font-size:12px;font-weight:500;cursor:pointer;font-family:'Outfit',sans-serif;transition:all .15s}
.fb:hover{border-color:#00d4ff40;color:#00d4ff}.fb.on{background:#00d4ff20;color:#00d4ff;border-color:#00d4ff60}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px;padding:0 32px 40px}
.card{background:#0c1221;border:1px solid #1a2540;border-radius:16px;overflow:hidden;transition:border-color .2s}
.card:hover{border-color:#00d4ff30}
.card-stripe{height:3px}
.card-inner{padding:18px}
.model-badge-row{margin-bottom:12px}
.mbadge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.04em}
.mboth{background:#f59e0b15;color:#f59e0b;border:1px solid #f59e0b30}
.mclaude{background:#00d4ff10;color:#00d4ff;border:1px solid #00d4ff30}
.mgemini{background:#10b98115;color:#34d399;border:1px solid #10b98130}
.ch{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px}
.cl{display:flex;align-items:center;gap:12px}
.rn{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:#475569;background:#080d1a;border:1px solid #1a2540;border-radius:5px;padding:3px 8px}
.tn{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;color:#f1f5f9;line-height:1}
.tl{font-size:12px;color:#64748b;margin-top:4px}
.cr{text-align:right}
.vc{font-size:11px;font-weight:700;padding:4px 10px;border-radius:5px;display:inline-block;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.04em}
.ul{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;margin-top:5px}
.mg{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-bottom:12px}
.mb{background:#080d1a;border:1px solid #1a2540;border-radius:7px;padding:8px 9px}
.ml{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:3px}
.mv{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:600;color:#e2e8f0}
.nb{height:2px;background:#1a2540;border-radius:1px;margin-top:3px;overflow:hidden}
.rs{margin-bottom:12px}
.rt{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.07em;font-weight:700;margin-bottom:6px}
.rp{display:flex;flex-wrap:wrap;gap:3px}
.pill-pass{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:600;font-family:'JetBrains Mono',monospace;background:#10b98115;color:#34d399;border:1px solid #10b98125}
.pill-fail{font-size:10px;padding:2px 7px;border-radius:4px;font-weight:600;font-family:'JetBrains Mono',monospace;background:#1a2540;color:#334155;border:1px solid #1a2540}
.ag{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
.ab{display:flex;gap:12px;padding:12px 14px;border-radius:0 10px 10px 0;align-items:flex-start;border-left:2px solid;border-right:none;border-top:none;border-bottom:none}
.bull{background:#0f2318;border-left-color:#10b981}
.bear{background:#1f0d12;border-left-color:#f43f5e}
.cat{background:#0d1228;border-left-color:#3b82f6}
.tech-row{background:#13102a;border-left:2px solid #8b5cf6;display:flex;gap:12px;padding:12px 14px;border-radius:0 10px 10px 0;align-items:flex-start}
.ai{font-size:15px;font-weight:700;width:18px;flex-shrink:0;margin-top:1px}
.bull .ai{color:#10b981}.bear .ai{color:#f43f5e}.cat .ai{color:#3b82f6}.tech-row .ai{color:#8b5cf6}
.al{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.bull .al{color:#10b981}.bear .al{color:#f43f5e}.cat .al{color:#3b82f6}.tech-row .al{color:#8b5cf6}
.at{font-size:16px;line-height:1.7;color:#cbd5e1}
.convrow{display:flex;align-items:center;gap:8px;padding-top:10px;border-top:1px solid #1a2540}
.convlbl{font-size:11px;color:#475569;font-weight:500;white-space:nowrap}
.convbar{flex:1;height:3px;background:#1a2540;border-radius:2px;overflow:hidden}
.convnum{font-family:'JetBrains Mono',monospace;font-size:12px;color:#475569;font-weight:600}
.foot{padding:20px 32px;text-align:center;font-size:12px;color:#334155;border-top:1px solid #1a2540;background:#0c1221;margin-top:8px}
.foot a{color:#00d4ff;text-decoration:none;font-weight:500}
.xbtn{font-size:12px;color:#00d4ff;cursor:pointer;margin-top:6px;font-weight:600;display:inline-block}
.xbtn:hover{color:#67e8f9}
.xbox{font-size:15px;line-height:1.8;color:#94a3b8;margin-top:8px;padding-top:8px;border-top:1px dashed #1e2d45}
.prr{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#080d1a;border-radius:8px;margin-bottom:12px;border:1px solid #1a2540;flex-wrap:wrap}
.prl{font-size:11px;color:#64748b;font-weight:500}
.prlo{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#f43f5e}
.prsp{font-size:11px;color:#334155}
.prhi{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#10b981}
.prmi{font-family:'JetBrains Mono',monospace;font-size:11px;color:#475569;margin-left:auto;background:#0c1221;padding:2px 8px;border-radius:4px;border:1px solid #1a2540}
@media(max-width:640px){.grid{grid-template-columns:1fr;padding:0 16px 28px}.mg{grid-template-columns:repeat(3,1fr)}.sum{grid-template-columns:repeat(2,1fr);padding:16px}.header,.toolbar,.both-box{padding-left:16px;padding-right:16px}}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CANSLIM AI - {timestamp}</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<header class="header">
  <div class="lw"><div class="lm">CS</div><div class="lt">CAN<span>SLIM</span> AI Screener</div></div>
  <div class="hm"><span class="ld"></span>Claude + Gemini · canslimscreener.com<br><strong>{total}</strong> stocks screened · {timestamp}</div>
</header>
<div class="both-box">
  <span style="font-size:20px">&#11088;</span>
  <span class="both-label">Both AIs agree on:</span>
  {both_chips}
</div>
<div class="sum">
  <div class="sc"><div class="sv" style="color:#2563eb">{len(merged)}</div><div class="sl">Unique picks</div><div class="ss">{len(both_picks)} agreed by both</div></div>
  <div class="sc"><div class="sv" style="color:#d97706">{avg_n:,}</div><div class="sl">Avg N-Value</div><div class="ss">max 8,190</div></div>
  <div class="sc"><div class="sv" style="color:#16a34a">{avg_ep:+}%</div><div class="sl">Avg EPS Growth</div><div class="ss">quarterly YoY</div></div>
  <div class="sc"><div class="sv" style="color:#7c3aed">{avg_up:+}%</div><div class="sl">Avg AI Upside</div><div class="ss">1-2 year target</div></div>
</div>
<div class="toolbar">
  <span class="tbl">Filter:</span>
  <button class="fb on" onclick="f('all',this)">All picks</button>
  <button class="fb" onclick="f('both',this)">Both AIs &#11088;</button>
  <button class="fb" onclick="f('claude',this)">Claude only</button>
  <button class="fb" onclick="f('gemini',this)">Gemini only</button>
  <button class="fb" onclick="fv('STRONG BUY',this)">Strong Buy</button>
</div>
<div class="grid" id="grid">{cards}</div>
<div class="foot">
  Data: <a href="http://www.canslimscreener.com/" target="_blank">canslimscreener.com</a>
  &middot; Claude claude-sonnet-4-20250514 &middot; Gemini 3 Pro &middot; Not financial advice.
</div>
<script>
function xd(btn){{const b=btn.nextElementSibling;if(b.style.display==="none"){{b.style.display="block";btn.textContent=btn.textContent.replace("+ ","- ");}}else{{b.style.display="none";btn.textContent=btn.textContent.replace("- ","+ ");}}}}
function f(m,btn){{
  document.querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display=m==='all'?'':c.dataset.models===m?'':'none';
  }});
}}
function fv(v,btn){{
  document.querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display=c.dataset.verdict===v?'':'none';
  }});
}}
</script>
</body>
</html>"""


async def main():
    print("\nCANSLIM Screener - Claude + Gemini\n")
    print("Step 1/3 - Scraping...")
    stocks = await scrape()
    if not stocks:
        print("No stocks found.")
        sys.exit(1)
    print(f"Got {len(stocks)} stocks. Top 5: {', '.join(s['ticker'] for s in stocks[:5])}")

    print("\nStep 2/3 - AI analysis...")
    print(f"  Claude key set: {bool(CLAUDE_KEY)}")
    print(f"  Gemini key set: {bool(GEMINI_KEY)}")
    print("  [Claude]")
    claude_top = analyze_claude(stocks)
    print("  [Gemini] (top 10 stocks only - free tier limit)")
    gemini_top = analyze_gemini(stocks)

    merged = merge_picks(claude_top, gemini_top, stocks)
    both   = [s for s in merged if s.get("models") == "both"]
    if both:
        print(f"\n  Both AIs agree: {[s['ticker'] for s in both]}")

    print("\nStep 3/3 - Building dashboard...")
    ts = datetime.now().strftime("%B %d, %Y  %H:%M")
    OUTPUT.write_text(build_dashboard(merged, len(stocks), ts), encoding="utf-8")
    print(f"Saved to {OUTPUT.resolve()}")

    print("\nFinal picks (merged, no duplicates):")
    for s in merged:
        models = s.get("models","?")
        label  = "BOTH  " if models=="both" else "Claude" if models=="claude" else "Gemini"
        print(f"  [{label}] #{s.get('rank','?')} {s['ticker']:6s} {s.get('verdict','?'):15s} conv:{s.get('conviction',0)}/10")

    print("\nView: python3 -m http.server 8080")
    print("Ports tab -> 8080 -> globe -> /canslim_dashboard.html")

if __name__ == "__main__":
    asyncio.run(main())
