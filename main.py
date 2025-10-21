# main.py
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os
import time
import re
import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from pytrends.request import TrendReq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------- Config -----------------
SCRAPER_API_URL = (os.environ.get("SCRAPER_API_URL") or "").strip()
SERPAPI_KEY = (os.environ.get("SERPAPI_KEY") or "").strip()
PYTRENDS_TZ = int(os.environ.get("PYTRENDS_TZ", "0"))
DEFAULT_REGION = os.environ.get("DEFAULT_REGION", "worldwide")
DEFAULT_TIMEFRAME = os.environ.get("DEFAULT_TIMEFRAME", "today 5-y")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "30"))
PYTRENDS_DELAY_SEC = float(os.environ.get("PYTRENDS_DELAY_SEC", "2.0"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Simple in-memory cache (process local)
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SEC = 24 * 60 * 60  # 24h


def cache_get(key: str):
    row = CACHE.get(key)
    if not row:
        return None
    if (time.time() - row["ts"]) > CACHE_TTL_SEC:
        CACHE.pop(key, None)
        return None
    return row["val"]


def cache_set(key: str, val: Any):
    CACHE[key] = {"ts": time.time(), "val": val}


def make_cache_key(*parts) -> str:
    """Create a collision-resistant cache key"""
    combined = "::".join(str(p) for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()


def via_proxy(url: str) -> str:
    if not SCRAPER_API_URL:
        return url
    sep = "&" if "?" in SCRAPER_API_URL else "?"
    import urllib.parse as urlparse
    return f"{SCRAPER_API_URL}{sep}url={urlparse.quote(url, safe='')}"


# ----------------- Schemas -----------------
class AnalyzeOptions(BaseModel):
    region: str = Field(DEFAULT_REGION, description="Google Trends geo code, '' for worldwide")
    timeframe: str = Field(DEFAULT_TIMEFRAME, description="Pytrends timeframe, e.g., 'today 12-m'")
    include_supply: bool = True


class AnalyzeBody(BaseModel):
    keywords: List[str] = Field(..., min_items=1, max_items=25)
    options: AnalyzeOptions = AnalyzeOptions()


class DemandOut(BaseModel):
    current: int
    momentum_pct: float
    series: List[Dict[str, Any]]
    rising_queries: List[str]


class SupplyOut(BaseModel):
    spocket_count: Optional[int] = None
    zendrop_count: Optional[int] = None
    amazon_serp_estimate: Optional[int] = None
    aliexpress_serp_estimate: Optional[int] = None


class ScoreOut(BaseModel):
    opportunity: int
    label: str
    confidence: float


class KeywordResult(BaseModel):
    keyword: str
    demand: DemandOut
    supply: SupplyOut
    scores: ScoreOut


class AnalyzeResponse(BaseModel):
    batch_id: str
    region: str
    timeframe: str
    results: List[KeywordResult]
    generated_at: str
    cache_ttl_sec: int


# ----------------- Demand: Google Trends -----------------
_pytrends = None


def pt() -> TrendReq:
    global _pytrends
    if _pytrends is None:
        # Hardened session with retries and a real UA
        session_args = {
            "headers": {"User-Agent": USER_AGENT},
            "timeout": (5, 30),  # connect timeout, read timeout
        }
        _pytrends = TrendReq(
            hl="en-US",
            tz=PYTRENDS_TZ,
            requests_args=session_args,
            retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504)
            ),
        )
        # Add retry adapter to the underlying session as well
        s = _pytrends.requests
        s.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                )
            ),
        )
    return _pytrends


def fetch_trends(keyword: str, region: str, timeframe: str) -> Dict[str, Any]:
    key = make_cache_key("trends", keyword, region, timeframe)
    hit = cache_get(key)
    if hit:
        return hit

    default_out = {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}

    try:
        py = pt()
        geo = "" if region.lower() == "worldwide" else region
        py.build_payload([keyword], timeframe=timeframe, geo=geo)
        
        df = py.interest_over_time()
        if df is None or df.empty:
            cache_set(key, default_out)
            return default_out

        # Clean dataframe
        df = df.reset_index()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.rename(columns={keyword: "interest", "date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Calculate momentum (last 90d vs prior 90d) if enough points exist
        if len(df) >= 180:
            last = df.tail(90)["interest"].mean()
            prev = df.tail(180).head(90)["interest"].mean()
            momentum = ((last - prev) / prev * 100.0) if prev > 0 else 0.0
        elif len(df) >= 30:
            last = df.tail(30)["interest"].mean()
            prev = df.tail(60).head(30)["interest"].mean()
            momentum = ((last - prev) / prev * 100.0) if prev > 0 else 0.0
        else:
            momentum = 0.0

        current = int(df["interest"].iloc[-1])
        series = [{"t": d.isoformat(), "v": int(v)} for d, v in zip(df["date"], df["interest"])]

        # Rising queries
        rising: List[str] = []
        try:
            rq = py.related_queries() or {}
            data = rq.get(keyword, {})
            if data and data.get("rising") is not None and not data["rising"].empty:
                rising = [str(x) for x in data["rising"]["query"].head(5).tolist()]
        except Exception as e:
            print(f"Rising queries error for '{keyword}': {e}")
            rising = []

        out = {
            "current": current,
            "momentum_pct": float(round(momentum, 2)),
            "series": series,
            "rising_queries": rising
        }
        cache_set(key, out)
        return out

    except Exception as e:
        print(f"Trends fetch error for '{keyword}': {e}")
        cache_set(key, default_out)
        return default_out


async def fetch_trends_via_serpapi(keyword: str, region: str, timeframe: str) -> Dict[str, Any]:
    """
    Fallback Trends fetch via SerpAPI if direct PyTrends fails/returns empty.
    """
    if not SERPAPI_KEY:
        return {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}

    # Map 'today X-y' to SerpAPI's expected windows
    win = "12m"
    if "5-y" in timeframe:
        win = "5y"
    elif "3-m" in timeframe:
        win = "3m"
    elif "7-d" in timeframe:
        win = "7d"

    params = {
        "engine": "google_trends",
        "q": keyword,
        "data_type": "TIMESERIES",
        "hl": "en-US",
        "geo": "" if region.lower() == "worldwide" else region,
        "date": win,
        "api_key": SERPAPI_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            r = await client.get("https://serpapi.com/search.json", params=params)
            if r.status_code != 200:
                return {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}
            j = r.json()

            # Parse time series
            points = j.get("interest_over_time", []) or j.get("timeline", [])
            series = []
            vals = []
            for p in points:
                t = p.get("date") or p.get("formattedTime")
                v_raw = p.get("value")
                if isinstance(v_raw, list) and v_raw:
                    v = v_raw[0]
                else:
                    v = v_raw or 0
                v = int(v or 0)
                series.append({"t": str(t), "v": v})
                vals.append(v)

            current = int(vals[-1]) if vals else 0

            # Momentum: last 90 vs previous 90 (or 30 vs previous 30 if shorter)
            def avg(lst):
                return sum(lst) / len(lst) if lst else 0
            if len(vals) >= 180:
                last = avg(vals[-90:])
                prev = avg(vals[-180:-90])
            elif len(vals) >= 60:
                last = avg(vals[-30:])
                prev = avg(vals[-60:-30])
            else:
                last = prev = 0
            mom = ((last - prev) / prev * 100.0) if prev > 0 else 0.0

            # Rising queries
            rq = j.get("rising_queries", {}).get("rising", []) or []
            rising = [str(x.get("query")) for x in rq[:5] if x.get("query")]

            return {
                "current": current,
                "momentum_pct": float(round(mom, 2)),
                "series": series,
                "rising_queries": rising,
            }
    except Exception:
        return {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}

# ----------------- Supply: Spocket & Zendrop (public mirrors) -----------------
CARD_RE = re.compile(
    r"(Add\s+to\s+(?:import\s+list|cart)|In\s+Stock|Ships\s+from|Trending|SKU)",
    re.I
)


def count_matches(text: str) -> int:
    """Count product card-ish markers; fallback to number of price-like patterns"""
    count = len(CARD_RE.findall(text))
    if count < 3:
        prices = re.findall(
            r"([$€£R]|ZAR)\s*\$?\s*([\d]{1,3}(?:[ ,]\d{3})*(?:\.\d{1,2})?)",
            text,
            flags=re.I
        )
        count = max(count, len(prices))
    return count


async def spocket_count(client: httpx.AsyncClient, kw: str) -> Optional[int]:
    import urllib.parse as urlparse
    urls = [
        f"https://r.jina.ai/https://spocket.co/search?query={urlparse.quote(kw, safe='')}",
        f"https://r.jina.ai/https://spocket.co/?s={urlparse.quote(kw, safe='')}",
        f"https://r.jina.ai/https://spocket.co/collections/all?view=search&q={urlparse.quote(kw, safe='')}",
    ]
    for u in urls:
        try:
            r = await client.get(u)
            if r.status_code == 200 and r.text:
                c = count_matches(r.text)
                if c:
                    return c
        except Exception as e:
            print(f"Spocket error for '{kw}': {e}")
            continue
    return None


async def zendrop_count(client: httpx.AsyncClient, kw: str) -> Optional[int]:
    """Best-effort public page (text mirror). If DOM changes, gracefully returns None."""
    import urllib.parse as urlparse
    urls = [
        f"https://r.jina.ai/https://app.zendrop.com/search?q={urlparse.quote(kw, safe='')}",
        f"https://r.jina.ai/https://www.zendrop.com/search?q={urlparse.quote(kw, safe='')}",
    ]
    for u in urls:
        try:
            r = await client.get(u)
            if r.status_code == 200 and r.text:
                c = count_matches(r.text)
                if c:
                    return c
        except Exception as e:
            print(f"Zendrop error for '{kw}': {e}")
            continue
    return None


# ----------------- Competition proxies via SERP API (optional) -----------------
async def serp_estimate(client: httpx.AsyncClient, site: str, kw: str) -> Optional[int]:
    if not SERPAPI_KEY:
        return None
    
    params = {
        "engine": "google",
        "q": f'site:{site} "{kw}"',
        "api_key": SERPAPI_KEY,
        "num": "10"
    }
    
    try:
        r = await client.get("https://serpapi.com/search.json", params=params)
        if r.status_code == 200:
            data = r.json()
            # SerpAPI: try search_information.total_results or search_metadata.total_results
            est = (
                data.get("search_information", {}).get("total_results")
                or data.get("search_metadata", {}).get("total_results")
            )
            if isinstance(est, int):
                return est
    except Exception as e:
        print(f"SERP API error for {site} '{kw}': {e}")
    return None


# ----------------- Scoring -----------------
def normalize_0_100(vals: List[Optional[float]]) -> List[float]:
    xs = [0.0 if v is None else float(v) for v in vals]
    maxv, minv = max(xs), min(xs)
    if maxv == minv:
        return [0.0 for _ in xs]
    return [(x - minv) / (maxv - minv) * 100.0 for x in xs]


def label_from_score(s: int) -> str:
    if s >= 80:
        return "Hot Opportunity"
    if s >= 60:
        return "Good Potential"
    if s >= 40:
        return "Moderate"
    return "Saturated"


def compute_scores(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute opportunity scores for a batch of keyword results"""
    # Build arrays for normalization
    demand_curr = [b["demand"]["current"] for b in batch]
    demand_mom = [b["demand"]["momentum_pct"] for b in batch]
    supply_sp = [(b["supply"]["spocket_count"] or 0) for b in batch]
    supply_zd = [(b["supply"]["zendrop_count"] or 0) for b in batch]
    amazon_est = [b["supply"]["amazon_serp_estimate"] for b in batch]
    ali_est = [b["supply"]["aliexpress_serp_estimate"] for b in batch]
    rising_cnt = [len(b["demand"]["rising_queries"]) for b in batch]
    
    # Calculate confidence based on data availability
    confidence = []
    for b in batch:
        avail = 1  # trends always attempted
        if b["supply"]["spocket_count"] is not None:
            avail += 1
        if b["supply"]["zendrop_count"] is not None:
            avail += 1
        if b["supply"]["amazon_serp_estimate"] is not None:
            avail += 1
        if b["supply"]["aliexpress_serp_estimate"] is not None:
            avail += 1
        confidence.append(avail / 5.0 * 100.0)

    # Normalize demand (blend current + momentum)
    d_curr_norm = normalize_0_100(demand_curr)
    d_mom_norm = normalize_0_100(demand_mom)
    demand_score = [0.5 * dc + 0.5 * dm for dc, dm in zip(d_curr_norm, d_mom_norm)]

    # Supply/competition: lower is better (gap)
    # Build a composite "supply pressure" = max(normalized spocket, zendrop, amazon, ali)
    sp_norm = normalize_0_100(supply_sp)
    zd_norm = normalize_0_100(supply_zd)
    am_norm = normalize_0_100([0 if v is None else v for v in amazon_est])
    al_norm = normalize_0_100([0 if v is None else v for v in ali_est])
    supply_pressure = [max(a, b, c, d) for a, b, c, d in zip(sp_norm, zd_norm, am_norm, al_norm)]
    supply_gap = [100.0 - p for p in supply_pressure]

    # Rising queries bonus: normalized to 0-100
    rising_bonus = normalize_0_100(rising_cnt)

    # Final score weights (explainable, you can tweak):
    # 45% demand, 35% supply gap, 10% rising queries, 10% confidence
    out_scores = []
    for i in range(len(batch)):
        score = (
            0.45 * demand_score[i] +
            0.35 * supply_gap[i] +
            0.10 * rising_bonus[i] +
            0.10 * confidence[i]
        )
        s_int = int(round(score))
        out_scores.append({
            "opportunity": s_int,
            "label": label_from_score(s_int),
            "confidence": round(confidence[i] / 100.0, 2),
        })
    return out_scores


# ----------------- FastAPI -----------------
app = FastAPI(title="Trend & Demand Data API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeBody):
    keywords = [k.strip() for k in body.keywords if k and k.strip()]
    if not (1 <= len(keywords) <= 25):
        raise HTTPException(status_code=400, detail="Provide 1..25 keywords")

    region = body.options.region
    timeframe = body.options.timeframe

    # Demand (Google Trends) — serial is fine; pytrends doesn't love many threads
    demand_map: Dict[str, Dict[str, Any]] = {}
    for kw in keywords:
        d = fetch_trends(kw, region, timeframe)
        # Fallback to SerpAPI Trends if PyTrends returned empty and a key is present
        if SERPAPI_KEY and (not d.get("series")):
            d = await fetch_trends_via_serpapi(kw, region, timeframe)
        demand_map[kw] = d
        time.sleep(PYTRENDS_DELAY_SEC)  # configurable delay

    # Supply (Spocket/Zendrop + SERP proxies) — async for speed
    supply_map: Dict[str, Dict[str, Any]] = {}

    if body.options.include_supply:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True
        ) as client:
            # Create all tasks upfront for true parallel execution
            all_tasks = []
            task_map = {}  # Track which task belongs to which keyword/source

            for kw in keywords:
                # Initialize supply data structure
                supply_map[kw] = {
                    "spocket_count": None,
                    "zendrop_count": None,
                    "amazon_serp_estimate": None,
                    "aliexpress_serp_estimate": None,
                }

                # Spocket
                t_sp = spocket_count(client, kw)
                all_tasks.append(t_sp)
                task_map[id(t_sp)] = (kw, "spocket")

                # Zendrop
                t_zd = zendrop_count(client, kw)
                all_tasks.append(t_zd)
                task_map[id(t_zd)] = (kw, "zendrop")

                # Amazon (if API key available)
                if SERPAPI_KEY:
                    t_am = serp_estimate(client, "amazon.com", kw)
                    all_tasks.append(t_am)
                    task_map[id(t_am)] = (kw, "amazon")

                # AliExpress (if API key available)
                if SERPAPI_KEY:
                    t_al = serp_estimate(client, "aliexpress.com", kw)
                    all_tasks.append(t_al)
                    task_map[id(t_al)] = (kw, "aliexpress")

            # Execute ALL tasks in parallel
            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Map results back to keywords
            for i, result in enumerate(results):
                task_id = id(all_tasks[i])
                kw, source = task_map[task_id]

                # Handle exceptions gracefully
                value = None if isinstance(result, Exception) else result

                if source == "spocket":
                    supply_map[kw]["spocket_count"] = value
                elif source == "zendrop":
                    supply_map[kw]["zendrop_count"] = value
                elif source == "amazon":
                    supply_map[kw]["amazon_serp_estimate"] = value
                elif source == "aliexpress":
                    supply_map[kw]["aliexpress_serp_estimate"] = value
    else:
        # No supply data requested
        for kw in keywords:
            supply_map[kw] = {
                "spocket_count": None,
                "zendrop_count": None,
                "amazon_serp_estimate": None,
                "aliexpress_serp_estimate": None,
            }

    # Build batch & score
    batch: List[Dict[str, Any]] = []
    for kw in keywords:
        batch.append({
            "keyword": kw,
            "demand": demand_map[kw],
            "supply": supply_map[kw],
        })

    scores = compute_scores(batch)
    results: List[KeywordResult] = []
    for row, sc in zip(batch, scores):
        results.append(KeywordResult(
            keyword=row["keyword"],
            demand=DemandOut(**row["demand"]),
            supply=SupplyOut(**row["supply"]),
            scores=ScoreOut(**sc),
        ))

    # Sort by opportunity score descending
    results = sorted(results, key=lambda r: r.scores.opportunity, reverse=True)

    return AnalyzeResponse(
        batch_id=f"b_{int(time.time())}",
        region=region,
        timeframe=timeframe,
        results=results,
        generated_at=datetime.utcnow().isoformat() + "Z",
        cache_ttl_sec=CACHE_TTL_SEC,
    )