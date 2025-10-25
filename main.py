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
import json
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
from openai import OpenAI

# ----------------- Config -----------------
SCRAPER_API_URL = (os.environ.get("SCRAPER_API_URL") or "").strip()
SERPAPI_KEY = (os.environ.get("SERPAPI_KEY") or "").strip()
PYTRENDS_TZ = int(os.environ.get("PYTRENDS_TZ", "0"))
DEFAULT_REGION = os.environ.get("DEFAULT_REGION", "worldwide")
DEFAULT_TIMEFRAME = os.environ.get("DEFAULT_TIMEFRAME", "today 5-y")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "30"))
# PYTRENDS_DELAY_SEC: How long to wait between Google Trends requests (PyTrends)
PYTRENDS_DELAY_SEC = float(os.environ.get("PYTRENDS_DELAY_SEC", "2.0"))
# Optional debug logging flag for quieter logs in production
DEBUG_LOGS = (os.environ.get("DEBUG_LOGS", "false").strip().lower() in ("1", "true", "yes"))
def log(msg: str):
    if DEBUG_LOGS:
        print(msg)
# Allow forcing SerpAPI for demand (for debugging/fallback)
FORCE_SERPAPI_TRENDS = (os.environ.get("FORCE_SERPAPI_TRENDS", "false").strip().lower() in ("1","true","yes"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# --- Seeds CSV config ---
SEEDS_CSV_PATH = os.environ.get("SEEDS_CSV_PATH", "./data/seeds.csv").strip()

# ---- SerpAPI timeframe mapping ----
def _serpapi_date_from_timeframe(tf: str) -> str:
    tf = (tf or "").strip().lower()
    if tf.startswith("today ") or tf.startswith("now "):
        return tf
    if "5-y" in tf:
        return "today 5-y"
    if "12-m" in tf or "1-y" in tf:
        return "today 12-m"
    if "3-m" in tf:
        return "today 3-m"
    if "1-m" in tf:
        return "today 1-m"
    if "7-d" in tf or "now 7" in tf:
        return "now 7-d"
    return "today 12-m"

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
# ----------------- Demand: Google Trends -----------------
_pytrends = None

def pt() -> TrendReq:
    """
    Create a TrendReq without pytrends' internal Retry (retries=0).
    We'll handle retries ourselves around the calls, which avoids the
    urllib3 v2 method_whitelist incompatibility.
    """
    global _pytrends
    if _pytrends is None:
        session_args = {
            "headers": {"User-Agent": USER_AGENT},
        }
        _pytrends = TrendReq(
            hl="en-US",
            tz=PYTRENDS_TZ,
            timeout=(5, 30),   # (connect, read)
            requests_args=session_args,
            retries=0,         # IMPORTANT: prevent pytrends from building Retry(method_whitelist=...)
            backoff_factor=0.0 # ignored when retries=0
        )
    return _pytrends


def fetch_trends(keyword: str, region: str, timeframe: str) -> Dict[str, Any]:
    key = make_cache_key("trends", keyword, region, timeframe)
    hit = cache_get(key)
    if hit:
        return hit

    default_out = {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}

    py = pt()
    geo = "" if region.lower() == "worldwide" else region

    # Our own tiny retry loop to replace pytrends' internal Retry
    attempts = 3
    backoff = 0.6

    for attempt in range(1, attempts + 1):
        try:
            py.build_payload([keyword], timeframe=timeframe, geo=geo)

            df = py.interest_over_time()
            if df is None or df.empty:
                cache_set(key, default_out)
                return default_out

            df = df.reset_index()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df = df.rename(columns={keyword: "interest", "date": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.date

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

            # Rising queries (best-effort; don't fail the whole call)
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
            print(f"Trends fetch error (attempt {attempt}/{attempts}) for '{keyword}': {e}")
            if attempt < attempts:
                time.sleep(backoff)
                backoff *= 1.5
            else:
                cache_set(key, default_out)
                return default_out



async def fetch_trends_via_serpapi_retry(keyword: str, region: str, timeframe: str) -> Dict[str, Any]:
    """
    Try SerpAPI Trends with A/B/C patterns and print logs like earlier:
      A) TIMESERIES + hl=en
      B) TIMESERIES (no hl)
      C) No data_type, no hl
    """
    empty = {"current": 0, "momentum_pct": 0.0, "series": [], "rising_queries": []}
    if not SERPAPI_KEY:
        return empty

    date_str = _serpapi_date_from_timeframe(timeframe)
    geo = "" if not region or region.lower() in ("", "worldwide") else region

    async def _try(params: dict, tag: str) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
                r = await client.get("https://serpapi.com/search.json", params=params)
                print(f"📊 SerpAPI {tag} code={r.status_code} for '{keyword}' date={params.get('date')} geo={params.get('geo','')}")
                if r.status_code != 200:
                    body = (r.text or "")[:300]
                    print(
                        f"ℹ️ {tag}: SerpAPI non-200 for '{keyword}' "
                        f"code={r.status_code} params={{date:{params.get('date')}, geo:{params.get('geo')}}} body={body}"
                    )
                    return None
                j = r.json()
                iot = j.get("interest_over_time", {}) or {}
                points = iot.get("timeline_data", []) or iot.get("timelineData", []) or []
                if not points:
                    print(f"⚠️ Empty timeline data for '{keyword}' on {tag}")
                    return None

                series, vals = [], []
                for p in points:
                    t = p.get("date") or p.get("formattedTime") or ""
                    vs = p.get("values") or p.get("value") or []
                    v = 0
                    if isinstance(vs, list) and vs:
                        v0 = vs[0]
                        if isinstance(v0, dict):
                            v_raw = v0.get("value") or v0.get("extracted_value") or 0
                        else:
                            v_raw = v0
                        if isinstance(v_raw, str):
                            v = 0 if "<" in v_raw else (int(v_raw) if v_raw.isdigit() else 0)
                        else:
                            v = int(v_raw or 0)
                    series.append({"t": str(t), "v": v})
                    vals.append(v)

                def avg(lst):
                    return sum(lst) / len(lst) if lst else 0
                if len(vals) >= 180:
                    last, prev = avg(vals[-90:]), avg(vals[-180:-90])
                elif len(vals) >= 60:
                    last, prev = avg(vals[-30:]), avg(vals[-60:-30])
                else:
                    last = avg(vals[-14:]) if len(vals) >= 28 else avg(vals)
                    prev = avg(vals[-28:-14]) if len(vals) >= 28 else 0
                mom = ((last - prev) / prev * 100.0) if prev > 0 else 0.0

                print(f"✅ SerpAPI trends OK for '{keyword}': points={len(series)} current={vals[-1] if vals else 0}")
                return {
                    "current": int(vals[-1]) if vals else 0,
                    "momentum_pct": float(round(mom, 2)),
                    "series": series,
                    "rising_queries": [],
                }
        except Exception as e:
            print(
                f"❌ SerpAPI trends exception on {tag} for '{keyword}': {repr(e)} "
                f"params={{date:{params.get('date')}, geo:{params.get('geo')}}}"
            )
            return None

    # A) TIMESERIES + hl
    params_a = {
        "engine": "google_trends",
        "q": keyword,
        "data_type": "TIMESERIES",
        "hl": "en",
        "date": date_str,
        "api_key": SERPAPI_KEY,
    }
    if geo:
        params_a["geo"] = geo
    res = await _try(params_a, "A (TIMESERIES)")
    if res:
        return res

    # B) TIMESERIES without hl
    params_b = dict(params_a)
    params_b.pop("hl", None)
    res = await _try(params_b, "B (TIMESERIES no hl)")
    if res:
        return res

    # C) No data_type and no hl
    params_c = {k: v for k, v in params_a.items() if k not in ("data_type", "hl")}
    res = await _try(params_c, "C (no data_type/no hl)")
    if res:
        return res

    print(f"❌ SerpAPI trends failed after A/B/C for '{keyword}' — falling back to PyTrends")
    return fetch_trends(keyword, region, timeframe)

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
# ----------------- FastAPI -----------------
app = FastAPI(title="Trend & Demand Data API", version="1.0.0")

# CORS configuration (env-driven, with Vercel preview support)
# Accept explicit origins via ALLOWED_ORIGINS (comma-separated),
# plus an optional regex via ALLOWED_ORIGIN_REGEX (e.g., r"https://.*\\.vercel\\.app$")
_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_origins_list = [o.strip() for o in _origins_env.split(",") if o.strip()]
# Always include localhost:3000 for local dev unless explicitly disallowed
if not any("localhost:3000" in o for o in _origins_list):
    _origins_list.append("http://localhost:3000")
# Default regex: allow any Vercel preview/prod domain
_origin_regex = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app$")
# If you truly want wildcard, set ALLOWED_ORIGIN_REGEX to ".*" and ALLOWED_ORIGINS to empty.
# With regex, we can safely allow credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_origin_regex=_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

class ExplainBody(BaseModel):
    keyword: str
    demand: Dict[str, Any]
    supply: Dict[str, Any]
    scores: Dict[str, Any]
    tone: Optional[str] = "friendly"
    audience: Optional[str] = "beginner"


from fastapi import Request  # make sure Request is imported

@app.post("/explain")
async def explain(body: ExplainBody, request: Request):
    # Build cache key
    ck = make_cache_key(
        "explain",
        body.keyword,
        json.dumps(body.demand, sort_keys=True),
        json.dumps(body.supply, sort_keys=True),
        json.dumps(body.scores, sort_keys=True),
        body.tone, body.audience
    )

    # Allow cache bypass via header for billing/diagnostics
    bypass_cache = request.headers.get("x-bypass-cache") == "1"
    if not bypass_cache:
        hit = cache_get(ck)
        if hit:
            return {"explanation": hit, "source": "cache"}

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="LLM not configured (OPENAI_API_KEY missing)")

    client = OpenAI(api_key=OPENAI_API_KEY)
    print("Using OpenAI base URL:", getattr(client, "base_url", "<unknown>"))

    model = (OPENAI_MODEL or "gpt-5-mini").strip()
    is_gpt5 = model.startswith("gpt-5")

    system_msg = (
        "You are a helpful ecommerce analyst. "
        "Write 2–4 short sentences in plain language (<=60 words). "
        "Include exactly one concrete next step. Do not reveal chain-of-thought."
    )
    data_dict = {
        "keyword": body.keyword,
        "demand": body.demand,
        "supply": body.supply,
        "scores": body.scores,
    }
    user_prompt = (
        "Explain this product opportunity for a {aud} in a {tone} tone.\n\nDATA:\n{data}"
    ).format(
        aud=body.audience or "beginner",
        tone=body.tone or "friendly",
        data=json.dumps(data_dict, indent=2),
    )

    def extract_text(resp):
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def extract_usage(resp):
        try:
            u = getattr(resp, "usage", None)
            if not u:
                return None
            return {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
        except Exception:
            return None

    try:
        text = ""
        usage = None

        if is_gpt5:
            # GPT-5 mini: ONLY send max_completion_tokens (no temperature/top_p/penalties)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",  "content": user_prompt},
                ],
                max_completion_tokens=256,
            )
            text = extract_text(resp)
            usage = extract_usage(resp)

            # If empty or length-stopped, try once more with a bit more room
            if not text or (resp.choices and resp.choices[0].finish_reason == "length"):
                resp2 = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",  "content": user_prompt},
                    ],
                    max_completion_tokens=384,
                )
                text = extract_text(resp2)
                usage = extract_usage(resp2)

        else:
            # Non-reasoning models (e.g., gpt-4o-mini)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",  "content": user_prompt},
                ],
                max_tokens=240,
                temperature=0.2,
            )
            text = extract_text(resp)
            usage = extract_usage(resp)

        # Fallback to a non-reasoning model if still empty
        if not text:
            fallback_model = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
            resp3 = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",  "content": user_prompt},
                ],
                max_tokens=240,
                temperature=0.2,
            )
            text = extract_text(resp3)
            usage = extract_usage(resp3)

        if not text:
            raise HTTPException(status_code=502, detail="LLM returned empty response")

        cache_set(ck, text)
        return {"explanation": text, "source": "llm", "usage": usage}

    except HTTPException:
        raise
    except Exception as e:
        print(f"LLM error for keyword '{body.keyword}': {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeBody):
    keywords = [k.strip() for k in body.keywords if k and k.strip()]
    if not (1 <= len(keywords) <= 25):
        raise HTTPException(status_code=400, detail="Provide 1..25 keywords")

    region = body.options.region
    timeframe = body.options.timeframe

    # Demand (Google Trends)
    demand_map: Dict[str, Dict[str, Any]] = {}
    for kw in keywords:
        if body.options.include_supply and SERPAPI_KEY:
            # Deep Mode: force SerpAPI A/B/C pattern and logs
            d = await fetch_trends_via_serpapi_retry(kw, region, timeframe)
        else:
            # Fast Mode: try PyTrends first, fallback to SerpAPI if empty
            d = fetch_trends(kw, region, timeframe)
            if SERPAPI_KEY and (not d.get("series")):
                d = await fetch_trends_via_serpapi_retry(kw, region, timeframe)
        demand_map[kw] = d
        await asyncio.sleep(PYTRENDS_DELAY_SEC if not body.options.include_supply else 0.2)

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

# ----------------- Top E-commerce Trends (Global by default) -----------------

# Global storage for curated products with live scores
CURATED_PRODUCTS = {
    "last_updated": None,
    "products": [],
    "is_updating": False,  # Prevent concurrent updates
}

# Product seed list - ALPHABETICAL (no bias, sorted by live scores)
# Mix of proven dropshipping winners across categories
DEFAULT_PRODUCT_SEEDS = [
    # 🏠 Home & Living
    ("air fryer", "Home"),
    ("robot vacuum", "Home"),
    ("led strip lights", "Home"),
    ("throw blanket", "Home"),
    ("diffuser", "Home"),
    ("humidifier", "Home"),
    ("essential oil diffuser", "Home"),
    ("desk organizer", "Home"),
    ("candle warmer", "Home"),
    ("storage bins", "Home"),
    ("wall mirror", "Home"),
    ("smart plug", "Home"),
    ("smart light bulb", "Home"),
    ("air purifier", "Home"),
    ("electric kettle", "Home"),
    ("space heater", "Home"),
    ("dehumidifier", "Home"),
    ("weighted blanket", "Home"),
    ("blackout curtains", "Home"),
    ("cordless vacuum", "Home"),
    ("standing desk", "Home"),
    ("monitor light bar", "Home"),
    ("magnetic window cleaner", "Home"),
    ("co2 monitor", "Home"),
    ("robot mop", "Home"),

    # 💪 Fitness & Health
    ("resistance bands", "Fitness"),
    ("yoga mat", "Fitness"),
    ("massage gun", "Fitness"),
    ("foam roller", "Fitness"),
    ("adjustable dumbbells", "Fitness"),
    ("protein shaker", "Fitness"),
    ("fitness tracker", "Fitness"),
    ("water bottle", "Fitness"),
    ("treadmill", "Fitness"),
    ("exercise bike", "Fitness"),
    ("rowing machine", "Fitness"),
    ("ankle weights", "Fitness"),
    ("pilates ring", "Fitness"),
    ("door pull up bar", "Fitness"),
    ("ab roller", "Fitness"),

    # 🧴 Beauty & Personal Care
    ("facial roller", "Beauty"),
    ("hair straightener", "Beauty"),
    ("curling iron", "Beauty"),
    ("makeup brush set", "Beauty"),
    ("nail drill", "Beauty"),
    ("face mask", "Beauty"),
    ("lip gloss", "Beauty"),
    ("eyelash curler", "Beauty"),
    ("hair dryer", "Beauty"),
    ("electric toothbrush", "Beauty"),
    ("ice roller", "Beauty"),
    ("hair waver", "Beauty"),
    ("laser hair removal", "Beauty"),
    ("scalp massager", "Beauty"),
    ("dermaplaning tool", "Beauty"),

    # 🐶 Pets
    ("dog harness", "Pets"),
    ("cat tree", "Pets"),
    ("automatic pet feeder", "Pets"),
    ("dog bed", "Pets"),
    ("pet camera", "Pets"),
    ("cat litter box", "Pets"),
    ("dog collar", "Pets"),
    ("pet grooming kit", "Pets"),
    ("pet stroller", "Pets"),
    ("interactive cat toy", "Pets"),
    ("dog car seat", "Pets"),
    ("slow feeder bowl", "Pets"),
    ("pet water fountain", "Pets"),

    # 💻 Electronics & Gadgets
    ("bluetooth speaker", "Electronics"),
    ("portable charger", "Electronics"),
    ("power bank", "Electronics"),
    ("wireless earbuds", "Electronics"),
    ("smart watch", "Electronics"),
    ("laptop stand", "Electronics"),
    ("phone case", "Electronics"),
    ("ring light", "Electronics"),
    ("gaming mouse", "Electronics"),
    ("keyboard", "Electronics"),
    ("monitor stand", "Electronics"),
    ("usb hub", "Electronics"),
    ("tablet stand", "Electronics"),
    ("action camera", "Electronics"),
    ("mini projector", "Electronics"),
    ("bluetooth tracker", "Electronics"),
    ("car phone holder", "Electronics"),
    ("magnetic power bank", "Electronics"),
    ("dash cam", "Electronics"),
    ("wifi extender", "Electronics"),

    # 🗂️ Office / Home Office
    ("laptop riser", "Office"),
    ("ergonomic mouse", "Office"),
    ("mechanical keyboard", "Office"),
    ("monitor riser", "Office"),
    ("desk pad", "Office"),
    ("cable management box", "Office"),
    ("document holder", "Office"),
    ("whiteboard", "Office"),
    ("standing desk mat", "Office"),
    ("office chair cushion", "Office"),

    # 👕 Fashion & Accessories
    ("crossbody bag", "Fashion"),
    ("sling bag", "Fashion"),
    ("tote bag", "Fashion"),
    ("sneakers", "Fashion"),
    ("hoodie", "Fashion"),
    ("leggings", "Fashion"),
    ("watch", "Fashion"),
    ("sunglasses", "Fashion"),
    ("hat", "Fashion"),
    ("jewelry box", "Fashion"),
    ("wallet", "Fashion"),
    ("beanie", "Fashion"),
    ("puffer jacket", "Fashion"),
    ("ankle boots", "Fashion"),
    ("thermal socks", "Fashion"),

    # 🎁 Gifts (evergreen)
    ("gift basket", "Gifts"),
    ("personalized necklace", "Gifts"),
    ("whiskey decanter set", "Gifts"),
    ("spa gift set", "Gifts"),
    ("coffee gift set", "Gifts"),
    ("custom photo frame", "Gifts"),

    # 👶 Baby & Kids
    ("baby monitor", "Baby"),
    ("stroller", "Baby"),
    ("baby carrier", "Baby"),
    ("diaper bag", "Baby"),
    ("baby bottle warmer", "Baby"),
    ("play mat", "Baby"),
    ("night light", "Baby"),
    ("crib mobile", "Baby"),
    ("white noise machine", "Baby"),
    ("baby lounger", "Baby"),

    # 🍳 Kitchen & Dining
    ("blender", "Kitchen"),
    ("coffee grinder", "Kitchen"),
    ("milk frother", "Kitchen"),
    ("knife set", "Kitchen"),
    ("cutting board", "Kitchen"),
    ("storage jars", "Kitchen"),
    ("kitchen scale", "Kitchen"),
    ("toaster oven", "Kitchen"),
    ("electric griddle", "Kitchen"),
    ("microwave", "Kitchen"),
    ("cast iron skillet", "Kitchen"),
    ("pressure cooker", "Kitchen"),
    ("immersion blender", "Kitchen"),
    ("air fryer liners", "Kitchen"),
    ("electric lunch box", "Kitchen"),

    # 🌿 Outdoors & Travel
    ("camping lantern", "Outdoors"),
    ("portable fan", "Outdoors"),
    ("tent", "Outdoors"),
    ("cooler bag", "Outdoors"),
    ("travel pillow", "Outdoors"),
    ("carry on luggage", "Outdoors"),
    ("sleeping bag", "Outdoors"),
    ("solar power bank", "Outdoors"),
    ("hammock", "Outdoors"),
    ("beach umbrella", "Outdoors"),
    ("packing cubes", "Outdoors"),
    ("inflatable paddle board", "Outdoors"),
    ("portable power station", "Outdoors"),

    # 🛠️ Home Improvement & Car
    ("rgb light bulbs", "Home"),
    ("cordless drill", "Home"),
    ("laser level", "Home"),
    ("security camera", "Home"),
    ("video doorbell", "Home"),
    ("tire inflator", "Automotive"),
    ("car vacuum", "Automotive"),
    ("trunk organizer", "Automotive"),
    ("sun shade", "Automotive"),
    ("windshield cover", "Automotive"),
    ("phone mount for car", "Automotive"),
    ("car seat covers", "Automotive"),
    ("led interior car lights", "Automotive"),
    ("tire pressure gauge", "Automotive"),
    ("car trash can", "Automotive"),
    ("portable jump starter", "Automotive"),
    ("roof cargo bag", "Automotive"),
    ("snow brush", "Automotive"),
]

# ---- Seasonal packs (auto-merged based on month) ----
SEASONAL_PACKS = [
    # 🎄 Christmas (Nov–Dec)
    {
        "months": [11, 12],
        "seeds": [
            ("christmas tree", "Seasonal"),
            ("christmas lights", "Seasonal"),
            ("advent calendar", "Seasonal"),
            ("christmas ornaments", "Seasonal"),
            ("stocking", "Seasonal"),
            ("gift wrap", "Seasonal"),
            ("ugly christmas sweater", "Fashion"),
        ],
    },
    # 🎃 Halloween (Sep–Oct)
    {
        "months": [9, 10],
        "seeds": [
            ("halloween decorations", "Seasonal"),
            ("halloween costume", "Seasonal"),
            ("pumpkin carving kit", "Seasonal"),
            ("spider web decorations", "Seasonal"),
        ],
    },
    # 🦃 Thanksgiving (Nov)
    {
        "months": [11],
        "seeds": [
            ("thanksgiving decorations", "Seasonal"),
            ("roasting pan", "Kitchen"),
            ("meat thermometer", "Kitchen"),
            ("table runner", "Home"),
        ],
    },
    # 🎆 New Year / Resolutions (Dec–Jan)
    {
        "months": [12, 1],
        "seeds": [
            ("weekly planner", "Office"),
            ("habit tracker journal", "Office"),
            ("water bottle with time marker", "Fitness"),
            ("jump rope", "Fitness"),
        ],
    },
    # 💘 Valentine’s Day (Jan–Feb)
    {
        "months": [1, 2],
        "seeds": [
            ("heart necklace", "Fashion"),
            ("rose bear", "Gifts"),
            ("chocolate gift box", "Gifts"),
            ("preserved roses", "Gifts"),
        ],
    },
    # 👩 Mother’s Day (May)
    {
        "months": [5],
        "seeds": [
            ("jewelry organizer", "Fashion"),
            ("scented candle", "Home"),
            ("bath bomb gift set", "Beauty"),
        ],
    },
    # 👨 Father’s Day (June)
    {
        "months": [6],
        "seeds": [
            ("beard trimmer", "Beauty"),
            ("whiskey stones", "Kitchen"),
            ("multitool", "Home"),
        ],
    },
    # 🏫 Back to School (Jul–Sep)
    {
        "months": [7, 8, 9],
        "seeds": [
            ("backpack", "Fashion"),
            ("lunch box", "Kitchen"),
            ("pencil case", "Office"),
            ("graphing calculator", "Electronics"),
        ],
    },
    # 🌸 Spring Cleaning (Mar–Apr)
    {
        "months": [3, 4],
        "seeds": [
            ("steam mop", "Home"),
            ("lint remover", "Home"),
            ("vacuum storage bags", "Home"),
            ("gardening tools", "Seasonal"),
            ("seed starter kit", "Seasonal"),
        ],
    },
    # ☀️ Summer / Outdoors (May–Aug)
    {
        "months": [5, 6, 7, 8],
        "seeds": [
            ("inflatable pool", "Outdoors"),
            ("bug zapper", "Outdoors"),
            ("patio string lights", "Outdoors"),
            ("cooling towel", "Seasonal"),
            ("picnic blanket", "Seasonal"),
        ],
    },
    # 🍂 Fall (Sep–Nov)
    {
        "months": [9, 10, 11],
        "seeds": [
            ("heated blanket", "Seasonal"),
            ("fall wreath", "Seasonal"),
        ],
    },
    # ❄️ Winter (Dec–Feb)
    {
        "months": [12, 1, 2],
        "seeds": [
            ("hand warmers", "Seasonal"),
            ("snow shovel", "Seasonal"),
        ],
    },
]

# --- Helper: Load seeds from CSV ---
def load_seeds_from_csv(path: str) -> List[tuple]:
    """
    Load seeds from a CSV file with columns: keyword, category.
    Returns a list of (keyword, category) tuples, deduped and cleaned.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Seeds] Failed to read CSV at {path}: {e}")
        return []
    if "keyword" not in df.columns or "category" not in df.columns:
        print(f"[Seeds] CSV missing required columns 'keyword' and 'category' at {path}")
        return []
    # Clean and filter
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["keyword"] != "") & (df["category"] != "")]
    # Deduplicate by (keyword, category) lowercased
    seen = set()
    out: List[tuple] = []
    for _, row in df.iterrows():
        k = row["keyword"]
        c = row["category"]
        key = (k.lower(), c.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((k, c))
    return out


def seasonal_seeds(now: Optional[datetime] = None) -> List[tuple]:
    """Return seasonal seed tuples active for the current month (UTC)."""
    if now is None:
        now = datetime.utcnow()
    month = now.month
    out: List[tuple] = []
    for pack in SEASONAL_PACKS:
        if month in pack.get("months", []):
            out.extend(pack.get("seeds", []))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for k, c in out:
        key = (k.lower(), c)
        if key not in seen:
            seen.add(key)
            deduped.append((k, c))
    return deduped

# --- Initialize PRODUCT_SEEDS from CSV if available, else use defaults ---
PRODUCT_SEEDS: List[tuple] = DEFAULT_PRODUCT_SEEDS
try:
    _loaded = load_seeds_from_csv(SEEDS_CSV_PATH)
    if _loaded:
        PRODUCT_SEEDS = _loaded
        print(f"[Seeds] Loaded {len(PRODUCT_SEEDS)} seeds from {SEEDS_CSV_PATH}")
    else:
        print("[Seeds] Using built-in default seeds")
except Exception as e:
    print(f"[Seeds] Error loading CSV seeds, using defaults: {e}")


async def update_product_scores_background():
    """
    Background task that updates product scores using real Google Trends data.
    Uses PyTrends (free) to avoid burning SerpAPI credits.
    """
    from datetime import datetime
    
    # Prevent concurrent updates
    if CURATED_PRODUCTS.get("is_updating"):
        print("[ProductUpdater] Update already in progress, skipping...")
        return CURATED_PRODUCTS.get("products", [])
    
    CURATED_PRODUCTS["is_updating"] = True
    
    try:
        print(f"[ProductUpdater] 🔄 Starting background update at {datetime.utcnow()}")

        region = "US"  # Use US as global proxy
        timeframe = "today 3-m"  # 3-month trend

        updated_products = []

        active_seeds = PRODUCT_SEEDS + seasonal_seeds()
        for idx, (keyword, category) in enumerate(active_seeds, 1):
            try:
                # Use PyTrends (FREE) - no API credits used
                demand = fetch_trends(keyword, region, timeframe)

                current = demand.get("current", 0)
                momentum = demand.get("momentum_pct", 0)
                rising_count = len(demand.get("rising_queries", []))

                # Calculate opportunity score (0-100)
                # Formula: Current interest is the main driver
                base_score = current  # Raw interest (0-100 from Google)
                momentum_bonus = max(-15, min(momentum * 0.3, 15))  # +/- 15 points max
                rising_bonus = min(rising_count * 1.5, 10)  # Up to 10 points

                score = max(0, min(100, base_score + momentum_bonus + rising_bonus))

                updated_products.append({
                    "keyword": keyword,
                    "category": category,
                    "score": round(score, 1),
                    "search_volume": current,
                    "momentum_pct": round(momentum, 1),
                    "rising_queries_count": rising_count,
                })

                print(f"[ProductUpdater] ({idx}/{len(active_seeds)}) {keyword}: score={score:.1f} (vol={current}, mom={momentum:.1f}%, rising={rising_count})")

                # Respect rate limits
                await asyncio.sleep(PYTRENDS_DELAY_SEC)

            except Exception as e:
                print(f"[ProductUpdater] ✗ Failed to update '{keyword}': {e}")
                # Add with zero score so we don't lose the keyword
                updated_products.append({
                    "keyword": keyword,
                    "category": category,
                    "score": 0,
                    "search_volume": 0,
                    "momentum_pct": 0,
                    "rising_queries_count": 0,
                })
                continue

        # Sort by score (HIGHEST FIRST - this is the key!)
        updated_products.sort(key=lambda x: x["score"], reverse=True)

        # Update global cache
        CURATED_PRODUCTS["products"] = updated_products
        CURATED_PRODUCTS["last_updated"] = datetime.utcnow().isoformat() + "Z"

        # Show top 5 for debugging
        print(f"[ProductUpdater] ✅ Top 5 products by score:")
        for i, p in enumerate(updated_products[:5], 1):
            print(f"  {i}. {p['keyword']}: {p['score']}")

        print(f"[ProductUpdater] ✅ Updated {len(updated_products)} products")
        return updated_products

    finally:
        CURATED_PRODUCTS["is_updating"] = False


async def ensure_products_updated():
    """
    Ensure products are updated. If stale (>2 hours) or empty, trigger update.
    """
    from datetime import datetime, timedelta
    
    last_updated = CURATED_PRODUCTS.get("last_updated")
    products = CURATED_PRODUCTS.get("products", [])
    
    # Check if we need to update
    needs_update = False
    
    if not products:
        print("[ProductUpdater] No products cached, triggering initial update")
        needs_update = True
    elif not last_updated:
        needs_update = True
    else:
        # Parse last update time
        try:
            last_update_time = datetime.fromisoformat(last_updated.replace("Z", ""))
            age = datetime.utcnow() - last_update_time
            
            # Update if older than 2 hours
            if age > timedelta(hours=2):
                print(f"[ProductUpdater] Data stale ({age.total_seconds()/3600:.1f}h old), updating...")
                needs_update = True
        except Exception as e:
            print(f"[ProductUpdater] Error parsing last_updated: {e}")
            needs_update = True
    
    if needs_update:
        # Trigger background update (non-blocking)
        asyncio.create_task(update_product_scores_background())
        
        # If we have stale data, return it immediately while updating in background
        if products:
            print("[ProductUpdater] Returning stale data while updating in background")
            return products
        else:
            # First time: wait for update to complete
            print("[ProductUpdater] First run: waiting for update to complete...")
            return await update_product_scores_background()
    
    return products



@app.get("/trends/categories")
async def get_categories():
    # Prefer currently computed products; if empty, derive from seeds + seasonal
    cats = set()
    if CURATED_PRODUCTS.get("products"):
        for p in CURATED_PRODUCTS["products"]:
            c = str(p.get("category", "")).strip()
            if c:
                cats.add(c)
    else:
        for _, c in PRODUCT_SEEDS + seasonal_seeds():
            if c:
                cats.add(c)
    return {"categories": sorted(cats)}


# --- Endpoint: reload seeds from CSV ---
@app.post("/trends/reload-seeds")
async def reload_seeds():
    """
    Reload seeds from SEEDS_CSV_PATH and trigger a background refresh.
    """
    global PRODUCT_SEEDS
    _loaded = load_seeds_from_csv(SEEDS_CSV_PATH)
    if _loaded:
        PRODUCT_SEEDS = _loaded
        msg = f"Reloaded {len(PRODUCT_SEEDS)} seeds from {SEEDS_CSV_PATH}"
    else:
        PRODUCT_SEEDS = DEFAULT_PRODUCT_SEEDS
        msg = "CSV missing/invalid — reverted to built-in defaults"
    # trigger a background refresh (non-blocking)
    asyncio.create_task(update_product_scores_background())
    return {"ok": True, "message": msg, "count": len(PRODUCT_SEEDS)}


@app.get("/trends/top-products")
async def get_top_products(geo: str = "GLOBAL", limit: int = 10, category: Optional[str] = None):
    """
    Returns top trending e-commerce products with LIVE scores.
    Scores auto-update every 2 hours.
    
    Data sources:
    - Google Trends (PyTrends) - FREE, updates every 2 hours
    - Curated product list - manually maintained
    
    Scoring formula:
    - Base: Current search interest (0-100)
    - Momentum: +/- 20 points based on 3-month trend
    - Rising queries: +10 points max for trending related searches
    """
    from datetime import datetime

    # Clamp limit
    try:
        limit = max(1, min(int(limit), 25))
    except Exception:
        limit = 10

    # Get products (will auto-update if stale)
    products = await ensure_products_updated()
    
    # If still no products (first run failed), return empty
    if not products:
        return {
            "geo": geo.upper(),
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "items": [],
            "last_data_update": None,
            "next_update": "In progress...",
            "error": "Product data is being fetched for the first time. Please try again in 30 seconds."
        }

    # Optional category filter (case-insensitive). Accepts "all" or empty as no filter.
    if category:
        cat_norm = category.strip().lower()
        if cat_norm not in ("", "all"):
            products = [p for p in products if str(p.get("category", "")).strip().lower() == cat_norm]
    
    # Take top N
    items = []
    for idx, product in enumerate(products[:limit], start=1):
        items.append({
            "rank": idx,
            "keyword": product["keyword"],
            "category": product["category"],
            "opportunity_score": product["score"],
            "search_volume": product.get("search_volume", 0),
            "momentum_pct": product.get("momentum_pct", 0),
            "source": "live_google_trends",
        })

    # Calculate next update time
    last_updated = CURATED_PRODUCTS.get("last_updated")
    next_update = "Within 2 hours"
    if last_updated:
        try:
            from datetime import datetime, timedelta
            last_time = datetime.fromisoformat(last_updated.replace("Z", ""))
            next_time = last_time + timedelta(hours=2)
            next_update = next_time.isoformat() + "Z"
        except Exception:
            pass

    result = {
        "geo": geo.upper(),
        "category": (category or "ALL").upper(),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "items": items,
        "last_data_update": last_updated,
        "next_update": next_update,
        "note": "Scores auto-update every 2 hours."
    }
    
    return result


@app.get("/trends/refresh")
async def force_refresh_products():
    """
    Admin endpoint to manually trigger a product score refresh.
    Useful for testing or forcing an update outside the 2-hour window.
    """
    print("[ProductUpdater] Manual refresh triggered")
    products = await update_product_scores_background()
    
    return {
        "status": "success",
        "updated_count": len(products),
        "updated_at": CURATED_PRODUCTS.get("last_updated"),
    }