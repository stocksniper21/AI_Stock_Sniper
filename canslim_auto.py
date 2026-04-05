#!/usr/bin/env python3
import asyncio, json, os, sys, time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright
import anthropic
import google.generativeai as genai

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
  "bull_case": "2 sentences.",
  "key_risk": "1 sentence.",
  "price_target_pct": 85,
  "catalyst": "next specific catalyst"
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
    return result[:TOP_N]


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
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")
    print(f"  Sending {len(stocks)} stocks to Gemini...")
    t0 = time.time()
    response = model.generate_content(build_prompt(stocks))
    print(f"  Gemini done in {time.time()-t0:.1f}s")
    result = parse_picks(response.text, stocks)
    print(f"  Gemini picks (>=7): {len(result)}")
    return result


def build_cards(top):
    if not top:
        return "<div style='padding:60px;text-align:center;color:#6b7280;font-size:14px'>No stocks passed conviction filter for this model.</div>"
    RULES  = ["sma50_150","sma150_200","span52","rs_rule","liquidity","above52h",
              "prevclose","sma200slope","inst_rule","abovesma50","sales_rule","eps_rule"]
    LABELS = ["SMA50>150","SMA150>200","52wkSpan","RS","Liquidity","Near52wH",
              "PrevClose","SMA200up","InstOwn",">SMA50","Sales","EPS"]
    vc = {"STRONG BUY":"#16a34a","BUY":"#2563eb","SPECULATIVE BUY":"#d97706","N/A":"#6b7280"}
    def pc(v):
        return "#16a34a" if v > 0 else "#dc2626" if v < 0 else "#6b7280"
    cards = ""
    for s in top:
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
        pills = ""
        for r, lb in zip(RULES, LABELS):
            cls = "pill-pass" if s.get(r) else "pill-fail"
            pills += f'<span class="{cls}">{lb}</span>'
        verdict   = s.get("verdict","N/A")
        ticker    = s.get("ticker","?")
        rank      = s.get("rank","?")
        theme     = s.get("theme","")
        bull      = s.get("bull_case","")
        risk      = s.get("key_risk","")
        catalyst  = s.get("catalyst","")
        cards += f"""
<div class="card" data-verdict="{verdict}">
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
    <div class="ab bull"><b class="ai">+</b><div><div class="al">Bull Case</div><div class="at">{bull}</div></div></div>
    <div class="ab bear"><b class="ai">-</b><div><div class="al">Key Risk</div><div class="at">{risk}</div></div></div>
    <div class="ab cat"><b class="ai">*</b><div><div class="al">Catalyst</div><div class="at">{catalyst}</div></div></div>
  </div>
  <div class="convrow">
    <span class="convlbl">Conviction</span>
    <div class="convbar"><div style="width:{conv*10}%;background:{cc};height:100%;border-radius:3px"></div></div>
    <span class="convnum">{conv}/10</span>
  </div>
</div>"""
    return cards


def stats(top):
    if not top:
        return 0, 0, 0, 0
    return (
        len(top),
        sum(s.get("nval",0) for s in top) // len(top),
        sum(s.get("epsPct",0) for s in top) // len(top),
        sum(s.get("price_target_pct",0) for s in top) // len(top)
    )


def build_dashboard(claude_top, gemini_top, total, timestamp):
    cc, can, caep, caup = stats(claude_top)
    gc, gan, gaep, gaup = stats(gemini_top)
    ct   = {s["ticker"] for s in claude_top}
    gt   = {s["ticker"] for s in gemini_top}
    both = ct & gt
    if both:
        both_html = " ".join(
            f'<span style="background:#fef3c7;color:#92400e;border:1px solid #fcd34d;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;font-family:monospace">{t}</span>'
            for t in sorted(both)
        )
    else:
        both_html = '<span style="color:#9ca3af;font-size:13px">No overlap between models yet</span>'
    cc_cards = build_cards(claude_top)
    gc_cards = build_cards(gemini_top)
    css = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f0f2f5;color:#111827;font-family:Inter,sans-serif}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:3px}
.header{background:#fff;border-bottom:1px solid #e5e7eb;padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:64px;position:sticky;top:0;z-index:100;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.lw{display:flex;align-items:center;gap:12px}
.lm{width:36px;height:36px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:12px;font-family:monospace}
.lt{font-size:16px;font-weight:700;color:#111827;letter-spacing:-.02em}.lt span{color:#2563eb}
.hm{font-size:12px;color:#6b7280;text-align:right;line-height:1.8}.hm strong{color:#111827}
.ld{display:inline-block;width:7px;height:7px;background:#16a34a;border-radius:50%;margin-right:5px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.both-box{margin:16px 32px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 20px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.both-label{font-size:12px;font-weight:600;color:#374151;white-space:nowrap}
.tabs{display:flex;padding:0 32px;border-bottom:2px solid #e5e7eb}
.tab{padding:12px 28px;font-size:14px;font-weight:600;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;color:#6b7280;transition:all .15s;display:flex;align-items:center;gap:8px}
.tab:hover{color:#111827}.tab.on{color:#2563eb;border-bottom-color:#2563eb}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600;background:#f3f4f6;color:#6b7280}
.tab.on .badge{background:#dbeafe;color:#1d4ed8}
.tab-content{display:none}.tab-content.on{display:block}
.sum{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:20px 32px}
.sc{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.sv{font-family:monospace;font-size:24px;font-weight:700;margin-bottom:4px}
.sl{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:500}
.ss{font-size:11px;color:#9ca3af;margin-top:2px}
.toolbar{padding:0 32px 16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tbl{font-size:12px;color:#6b7280;font-weight:500;margin-right:4px}
.fb{padding:5px 14px;border-radius:20px;border:1.5px solid #e5e7eb;background:#fff;color:#374151;font-size:12px;font-weight:500;cursor:pointer;transition:all .15s}
.fb:hover{border-color:#2563eb;color:#2563eb}.fb.on{background:#2563eb;color:#fff;border-color:#2563eb}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:16px;padding:0 32px 40px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.05);transition:box-shadow .2s,border-color .2s}
.card:hover{box-shadow:0 4px 16px rgba(0,0,0,.08);border-color:#d1d5db}
.ch{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px}
.cl{display:flex;align-items:center;gap:12px}
.rn{font-family:monospace;font-size:13px;font-weight:600;color:#9ca3af;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:3px 8px}
.tn{font-family:monospace;font-size:22px;font-weight:700;color:#111827;line-height:1}
.tl{font-size:12px;color:#6b7280;margin-top:4px}
.cr{text-align:right}
.vc{font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;display:inline-block;font-family:monospace}
.ul{font-family:monospace;font-size:18px;font-weight:700;margin-top:6px}
.mg{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:14px}
.mb{background:#f9fafb;border:1px solid #f3f4f6;border-radius:8px;padding:8px 10px}
.ml{font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:4px}
.mv{font-family:monospace;font-size:13px;font-weight:600;color:#111827}
.nb{height:3px;background:#e5e7eb;border-radius:2px;margin-top:4px;overflow:hidden}
.rs{margin-bottom:14px}
.rt{font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:6px}
.rp{display:flex;flex-wrap:wrap;gap:4px}
.pill-pass{font-size:10px;padding:3px 7px;border-radius:4px;font-weight:500;font-family:monospace;background:#dcfce7;color:#15803d;border:1px solid #bbf7d0}
.pill-fail{font-size:10px;padding:3px 7px;border-radius:4px;font-weight:500;font-family:monospace;background:#f9fafb;color:#9ca3af;border:1px solid #e5e7eb}
.ag{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
.ab{display:flex;gap:10px;padding:10px 12px;border-radius:8px;border:1px solid #e5e7eb;align-items:flex-start}
.bull{background:#f0fdf4;border-color:#bbf7d0}.bear{background:#fff1f2;border-color:#fecdd3}.cat{background:#eff6ff;border-color:#bfdbfe}
.ai{font-size:16px;font-weight:700;width:20px;flex-shrink:0;margin-top:1px}
.bull .ai{color:#16a34a}.bear .ai{color:#dc2626}.cat .ai{color:#2563eb}
.al{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#6b7280;margin-bottom:2px}
.at{font-size:12px;line-height:1.6;color:#374151}
.convrow{display:flex;align-items:center;gap:10px;padding-top:12px;border-top:1px solid #f3f4f6}
.convlbl{font-size:11px;color:#6b7280;font-weight:500;white-space:nowrap}
.convbar{flex:1;height:5px;background:#f3f4f6;border-radius:3px;overflow:hidden}
.convnum{font-family:monospace;font-size:12px;color:#374151;font-weight:600}
.foot{padding:20px 32px;text-align:center;font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;background:#fff;margin-top:8px}
.foot a{color:#2563eb;text-decoration:none;font-weight:500}
@media(max-width:640px){.grid{grid-template-columns:1fr;padding:0 16px 28px}.mg{grid-template-columns:repeat(3,1fr)}.sum{grid-template-columns:repeat(2,1fr);padding:16px}.header,.toolbar,.tabs,.both-box{padding-left:16px;padding-right:16px}}
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CANSLIM AI - {timestamp}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<header class="header">
  <div class="lw"><div class="lm">CS</div><div class="lt">CAN<span>SLIM</span> AI Screener</div></div>
  <div class="hm"><span class="ld"></span>Claude + Gemini &middot; canslimscreener.com<br><strong>{total}</strong> stocks screened &middot; {timestamp}</div>
</header>
<div class="both-box">
  <span style="font-size:18px">&#11088;</span>
  <span class="both-label">Both AIs agree on:</span>
  {both_html}
</div>
<div class="tabs">
  <div class="tab on" onclick="switchTab('claude',this)">Claude <span class="badge">{cc} picks</span></div>
  <div class="tab" onclick="switchTab('gemini',this)">Gemini <span class="badge">{gc} picks</span></div>
</div>
<div id="tab-claude" class="tab-content on">
  <div class="sum">
    <div class="sc"><div class="sv" style="color:#2563eb">{cc}</div><div class="sl">Claude Picks</div><div class="ss">conviction &ge; 7/10</div></div>
    <div class="sc"><div class="sv" style="color:#d97706">{can:,}</div><div class="sl">Avg N-Value</div><div class="ss">max 8190</div></div>
    <div class="sc"><div class="sv" style="color:#16a34a">{caep:+}%</div><div class="sl">Avg EPS Growth</div><div class="ss">quarterly YoY</div></div>
    <div class="sc"><div class="sv" style="color:#7c3aed">{caup:+}%</div><div class="sl">Avg AI Upside</div><div class="ss">1-2 year target</div></div>
  </div>
  <div class="toolbar">
    <span class="tbl">Filter:</span>
    <button class="fb on" onclick="fc('claude','all',this)">All</button>
    <button class="fb" onclick="fc('claude','STRONG BUY',this)">Strong Buy</button>
    <button class="fb" onclick="fc('claude','BUY',this)">Buy</button>
    <button class="fb" onclick="fc('claude','SPECULATIVE BUY',this)">Speculative</button>
  </div>
  <div class="grid" id="grid-claude">{cc_cards}</div>
</div>
<div id="tab-gemini" class="tab-content">
  <div class="sum">
    <div class="sc"><div class="sv" style="color:#16a34a">{gc}</div><div class="sl">Gemini Picks</div><div class="ss">conviction &ge; 7/10</div></div>
    <div class="sc"><div class="sv" style="color:#d97706">{gan:,}</div><div class="sl">Avg N-Value</div><div class="ss">max 8190</div></div>
    <div class="sc"><div class="sv" style="color:#16a34a">{gaep:+}%</div><div class="sl">Avg EPS Growth</div><div class="ss">quarterly YoY</div></div>
    <div class="sc"><div class="sv" style="color:#7c3aed">{gaup:+}%</div><div class="sl">Avg AI Upside</div><div class="ss">1-2 year target</div></div>
  </div>
  <div class="toolbar">
    <span class="tbl">Filter:</span>
    <button class="fb on" onclick="fc('gemini','all',this)">All</button>
    <button class="fb" onclick="fc('gemini','STRONG BUY',this)">Strong Buy</button>
    <button class="fb" onclick="fc('gemini','BUY',this)">Buy</button>
    <button class="fb" onclick="fc('gemini','SPECULATIVE BUY',this)">Speculative</button>
  </div>
  <div class="grid" id="grid-gemini">{gc_cards}</div>
</div>
<div class="foot">
  Data: <a href="http://www.canslimscreener.com/" target="_blank">canslimscreener.com</a>
  &middot; Claude claude-sonnet-4-20250514 &middot; Gemini 1.5 Pro &middot; Not financial advice.
</div>
<script>
function switchTab(n,el){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  document.getElementById('tab-'+n).classList.add('on');
}}
function fc(tab,v,btn){{
  const g=document.getElementById('grid-'+tab);
  g.closest('.tab-content').querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  g.querySelectorAll('.card').forEach(c=>{{
    c.style.display=v==='all'||c.dataset.verdict===v?'':'none';
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
    print("  [Claude]")
    claude_top = analyze_claude(stocks)
    print("  [Gemini]")
    gemini_top = analyze_gemini(stocks)
    both = {s["ticker"] for s in claude_top} & {s["ticker"] for s in gemini_top}
    if both:
        print(f"  Both agree: {sorted(both)}")
    print("\nStep 3/3 - Building dashboard...")
    ts = datetime.now().strftime("%B %d, %Y  %H:%M")
    OUTPUT.write_text(build_dashboard(claude_top, gemini_top, len(stocks), ts), encoding="utf-8")
    print(f"Saved to {OUTPUT.resolve()}")
    print("\nClaude picks:")
    for s in claude_top:
        print(f"  #{s.get('rank','?')} {s['ticker']:6s} {s.get('verdict','?'):15s} conv:{s.get('conviction',0)}/10")
    print("\nGemini picks:")
    for s in gemini_top:
        print(f"  #{s.get('rank','?')} {s['ticker']:6s} {s.get('verdict','?'):15s} conv:{s.get('conviction',0)}/10")
    print("\nView: python3 -m http.server 8080")
    print("Ports tab -> 8080 -> globe -> /canslim_dashboard.html")

if __name__ == "__main__":
    asyncio.run(main())
