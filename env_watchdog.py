# env_watchdog.py
from __future__ import annotations

import os
import json
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import requests
from openai import OpenAI
from tavily import TavilyClient


# -------------------------
# Categories / Topics
# -------------------------
TOPICS: Dict[str, str] = {
    "MARPOL Annex I – Oil": (
        "Monitor only recent changes, amendments, circulars or implementation guidance relevant "
        "to MARPOL Annex I (oil pollution). Focus strictly on what is new for ship operations."
    ),
    "MARPOL Annex II – Noxious Liquid Substances": (
        "Monitor only recent changes, amendments, circulars or implementation guidance affecting "
        "MARPOL Annex II (NLS). Include any new cargo categorisations, prewash requirements or "
        "key MEPC updates."
    ),
    "MARPOL Annex III – Harmful Substances in Packaged Form": (
        "Monitor only recent changes or cross-references between MARPOL Annex III and the IMDG Code "
        "that affect shipboard practice."
    ),
    "MARPOL Annex IV – Sewage": (
        "Monitor only recent developments on MARPOL Annex IV, including new special areas, updated "
        "discharge standards or equipment requirements."
    ),
    "MARPOL Annex V – Garbage": (
        "Monitor only recent changes related to MARPOL Annex V, such as updated garbage categories, "
        "record-keeping guidance, or new PSC focus areas."
    ),
    "MARPOL Annex VI – Air Pollution and GHG": (
        "Monitor only recent changes to MARPOL Annex VI, including SOx/NOx guidance, fuel sulphur "
        "enforcement, EEXI/CII developments, and IMO GHG strategy items. Include EU ETS / FuelEU only "
        "when they are official/regulatory developments."
    ),
    "Ballast Water Management Convention (BWM)": (
        "Monitor only recent changes for the BWM Convention, including D-1/D-2 implementation, "
        "sampling/inspection practices, or updated type-approval guidance."
    ),
    "Anti-fouling Systems (AFS Convention)": (
        "Monitor only recent changes under the AFS Convention, such as new substances controlled, "
        "updated guidelines or notable regional responses."
    ),
    "Regional / local regimes (EU, US, AUS, etc.)": (
        "Monitor only recent changes in major regional regimes affecting ships (EU ETS, FuelEU, MRV, "
        "US EPA, USCG, CARB, AMSA, etc.). Prefer regulator sources."
    ),
}

CATEGORY_TABS_ORDER = [
    "MARPOL Annex I – Oil",
    "MARPOL Annex II – Noxious Liquid Substances",
    "MARPOL Annex III – Harmful Substances in Packaged Form",
    "MARPOL Annex IV – Sewage",
    "MARPOL Annex V – Garbage",
    "MARPOL Annex VI – Air Pollution and GHG",
    "Ballast Water Management Convention (BWM)",
    "Anti-fouling Systems (AFS Convention)",
    "Regional / local regimes (EU, US, AUS, etc.)",
]

LOCAL_CATEGORY = "Regional / local regimes (EU, US, AUS, etc.)"

# -------------------------
# User-provided seed URLs (always read)
# - These will be fetched and included even if Tavily fails.
# - By default they are injected ONLY into LOCAL_CATEGORY.
# -------------------------
DEFAULT_EXTRA_URLS: List[str] = [
    "https://www.amsa.gov.au/about/regulations-and-standards/index-marine-notices",
    "https://ww2.eagle.org/en/rules-and-resources/regulatory-updates/regulatory-news.html",
    "https://www.dnv.com/maritime/technical-regulatory-news/",
    "https://www.classnk.or.jp/hp/en/tech_news.aspx",
    "https://www.ccs.org.cn/ccswzen/special?columnid=202206080248772574&id=0",
    "https://www.ccs.org.cn/ccswzen/columnList?columnid=202007171176731956",
    "https://www.ccs.org.cn/ccswzen/circularNotice?columnid=201900002000000071",
    "https://www.krs.co.kr/eng/",
    "https://decarbonization.krs.co.kr/eng/",
    "https://decarbonization.krs.co.kr/eng/Exclusive/Tech_ETC.aspx?MRID=973&URID=0",
    "https://www.bureauveritas.gr/newsroom",
]


# -------------------------
# Storage (merge-only; dedupe; cache)
# -------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
LATEST_RUN_PATH = os.path.join(DATA_DIR, "latest_run.json")
FETCH_CACHE_PATH = os.path.join(DATA_DIR, "fetch_cache.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedupe_items_keep_first_by_id(items: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        _id = it.get("id")
        if not _id:
            continue
        if _id in seen:
            continue
        seen.add(_id)
        out.append(it)
    return out


def load_state() -> dict:
    _ensure_data_dir()
    if not os.path.exists(STATE_PATH):
        return {"items": []}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        items = state.get("items", [])
        if not isinstance(items, list):
            state["items"] = []
            return state
        deduped = _dedupe_items_keep_first_by_id(items)
        if len(deduped) != len(items):
            state["items"] = deduped
            save_state(state)
        return state
    except Exception:
        return {"items": []}


def save_state(state: dict) -> None:
    _ensure_data_dir()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def save_latest_run(additions: List[dict]) -> None:
    _ensure_data_dir()
    payload = {"timestamp_utc": _utc_now_iso(), "additions": additions}
    with open(LATEST_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def load_latest_run() -> dict:
    _ensure_data_dir()
    if not os.path.exists(LATEST_RUN_PATH):
        return {"timestamp_utc": None, "additions": []}
    try:
        with open(LATEST_RUN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"timestamp_utc": None, "additions": []}


def _load_fetch_cache() -> dict:
    _ensure_data_dir()
    if not os.path.exists(FETCH_CACHE_PATH):
        return {}
    try:
        with open(FETCH_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_fetch_cache(cache: dict) -> None:
    _ensure_data_dir()
    with open(FETCH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=True, indent=2)


# -------------------------
# Clients
# -------------------------
def _tavily_client() -> TavilyClient:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing TAVILY_API_KEY environment variable.")
    return TavilyClient(api_key=key)


def _groq_client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)


# -------------------------
# Extra URL feeding (simple)
# -------------------------
def _parse_urls_text(raw: str) -> List[str]:
    """
    Parse newline- or comma-separated URLs from UI/env.
    Keeps https:// only, dedupes.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    parts: List[str] = []
    for line in raw.splitlines():
        parts.extend([p.strip() for p in line.split(",") if p.strip()])
    out: List[str] = []
    seen = set()
    for u in parts:
        if not u.startswith("https://"):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _sources_from_urls(urls: List[str], topic: str, score: float = 999.0) -> List[dict]:
    out: List[dict] = []
    for u in (urls or []):
        u = (u or "").strip()
        if not u.startswith("https://"):
            continue
        out.append(
            {
                "title": "User-provided URL",
                "url": u,
                "content": "",
                "score": score,
                "_topic": topic,
                "_seed": True,
            }
        )
    return out


def _dedupe_sources_by_url(sources: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for s in sources:
        u = (s.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(s)
    return out


# -------------------------
# Retrieval: search + fetch full page text
# -------------------------
def _search_topic(
    client: TavilyClient,
    topic_name: str,
    search_depth: str,
    max_results: int,
    preferred_domains: Optional[List[str]],
    window_days: int,
) -> List[dict]:
    q = (
        f"{topic_name} "
        f"(amendment OR circular OR resolution OR guideline OR enforcement OR delegated act OR regulation) "
        f"(IMO OR MEPC OR flag circular OR port state OR regulator OR class technical news) "
        f"last {window_days} days"
    )

    kwargs = {}
    if preferred_domains:
        # Ensure Tavily gets domains, not URLs
        clean = []
        for d in preferred_domains:
            d = (d or "").strip()
            if not d:
                continue
            d = d.replace("https://", "").replace("http://", "").strip("/")
            if d:
                clean.append(d)
        if clean:
            kwargs["include_domains"] = clean

    try:
        res = client.search(
            query=q,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_images=False,
            **kwargs,
        )

    except requests.exceptions.HTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        text = getattr(getattr(e, "response", None), "text", "")
        print(
            "TAVILY_HTTPERROR:",
            json.dumps(
                {
                    "topic": topic_name,
                    "status_code": status,
                    "response_text_head": (text or "")[:500],
                    "kwargs": kwargs,
                },
                ensure_ascii=True,
            ),
        )
        return []

    except Exception as e:
        print(
            "TAVILY_SEARCH_ERROR:",
            json.dumps(
                {
                    "topic": topic_name,
                    "error_type": type(e).__name__,
                    "error": str(e)[:300],
                    "kwargs": kwargs,
                },
                ensure_ascii=True,
            ),
        )
        return []

    results = res.get("results", []) if isinstance(res, dict) else []
    for r in results:
        r["_topic"] = topic_name
    return results


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url_text(
    url: str,
    cache: dict,
    cache_ttl_hours: int,
    timeout_sec: int,
    max_chars: int,
    polite_delay_sec: float,
) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        return ""

    now = datetime.now(timezone.utc)
    cached = cache.get(url)
    if isinstance(cached, dict):
        ts = cached.get("fetched_at_utc")
        body = cached.get("text", "")
        if ts and body:
            try:
                dt = datetime.fromisoformat(ts)
                age_hours = (now - dt).total_seconds() / 3600.0
                if age_hours <= float(cache_ttl_hours):
                    return str(body)[:max_chars]
            except Exception:
                pass

    if polite_delay_sec > 0:
        time.sleep(polite_delay_sec)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; EnvWatchdog/1.0; +https://example.invalid)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout_sec)
        if r.status_code >= 400:
            return ""
        content_type = (r.headers.get("content-type") or "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            # No PDF parsing here (avoids heavy deps).
            return ""

        html = r.text or ""
        text = _strip_html(html)
        text = text[:max_chars].strip()

        if text:
            cache[url] = {"fetched_at_utc": now.isoformat(timespec="seconds"), "text": text}
        return text
    except Exception:
        return ""


def _build_context_from_sources(
    sources: List[dict],
    fulltexts: Dict[str, str],
    max_total_chars: int,
) -> str:
    chunks: List[str] = []
    total = 0

    for s in sources:
        topic = (s.get("_topic") or "").strip()
        title = (s.get("title") or "").strip()
        url = (s.get("url") or "").strip()
        snippet = (s.get("content") or s.get("snippet") or "").strip()

        if not url:
            continue

        body = fulltexts.get(url) or ""
        if not body:
            body = snippet

        if not body:
            continue

        block = f"TOPIC: {topic}\nTITLE: {title}\nURL: {url}\nCONTENT:\n{body}\n"
        if total + len(block) > max_total_chars:
            break
        chunks.append(block)
        total += len(block)

    return "\n\n".join(chunks).strip()


# -------------------------
# LLM: rerank sources + extract structured updates
# -------------------------
RERANK_SYSTEM = (
    "You are selecting best primary sources for maritime environmental regulatory updates. "
    "Prefer official/primary sources (IMO, flags, regulators, class). Prefer items that are clearly "
    "recent within the provided window. Avoid blogs or low-quality sites."
)


def _safe_json_loads(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _rerank_sources(
    llm: OpenAI,
    model_id: str,
    topic: str,
    window_days: int,
    sources: List[dict],
    k: int,
) -> List[dict]:
    if not sources:
        return []

    mini = []
    for s in sources[:50]:
        mini.append(
            {
                "title": (s.get("title") or "")[:180],
                "url": s.get("url") or "",
                "snippet": (s.get("content") or s.get("snippet") or "")[:240],
                "score": s.get("score", None),
            }
        )

    prompt = (
        f"Topic: {topic}\n"
        f"Window: last {window_days} days\n"
        f"Select the best {k} sources (URLs) to extract regulatory updates.\n"
        f"Return ONLY a JSON array of URLs (strings). No other text.\n\n"
        f"Candidates:\n{json.dumps(mini, ensure_ascii=True)}"
    )

    try:
        resp = llm.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        arr = _safe_json_loads(raw)
        if isinstance(arr, list):
            wanted = [u for u in arr if isinstance(u, str) and u.startswith("http")]
            wanted_set = set(wanted)
            out = [s for s in sources if (s.get("url") or "") in wanted_set]
            out_sorted = []
            by_url = {s.get("url"): s for s in out}
            for u in wanted:
                if u in by_url:
                    out_sorted.append(by_url[u])
            return out_sorted[:k]
    except Exception:
        pass

    def _score(s):
        v = s.get("score")
        try:
            return float(v)
        except Exception:
            return 0.0

    return sorted(sources, key=_score, reverse=True)[:k]


EXTRACT_SYSTEM = (
    "You are an Environmental Specialist Watch Dog for an international ship management company. "
    "You extract only regulatory developments within a time window defined by the user. "
    "Prefer official sources (IMO, flags, regulators, class) and ignore older baseline rules.\n\n"
    "Return STRICT JSON only: a JSON array of objects. No markdown. No extra text.\n"
    "Each object must have keys:\n"
    "category, authority, instrument, date, summary, url\n"
    "date must be 'YYYY-MM-DD' if known; else 'date unclear'.\n"
    "summary MUST be exactly 2 lines (two sentences max total), practical and action-oriented: "
    "line 1 = operational impact; line 2 = documentation/survey/inspection action. "
    "Separate the two lines with a newline character (\\n). Do not use bullet symbols.\n"
    "url must be https or 'link unavailable'."
)


def _normalize_item(x: dict, fallback_category: str) -> Optional[dict]:
    if not isinstance(x, dict):
        return None

    category = (x.get("category") or fallback_category).strip() or fallback_category
    authority = (x.get("authority") or "").strip() or "authority unclear"
    instrument = (x.get("instrument") or "").strip() or "instrument unclear"
    date = (x.get("date") or "date unclear").strip() or "date unclear"
    summary = (x.get("summary") or "").strip() or "summary unclear"
    url = (x.get("url") or "link unavailable").strip() or "link unavailable"

    if not (url.startswith("https://") or url == "link unavailable"):
        url = "link unavailable"

    summary_lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    if len(summary_lines) >= 2:
        summary = summary_lines[0] + "\n" + summary_lines[1]
    elif len(summary_lines) == 1:
        summary = summary_lines[0] + "\n" + "Action: Review/implement and update documentation as applicable."
    else:
        summary = "summary unclear\nAction: Review/implement and update documentation as applicable."

    if (
        authority == "authority unclear"
        and instrument == "instrument unclear"
        and summary.startswith("summary unclear")
        and url == "link unavailable"
    ):
        return None

    return {
        "category": category,
        "authority": authority,
        "instrument": instrument,
        "date": date,
        "summary": summary,
        "url": url,
    }


def _item_id(item: dict) -> str:
    key = f"{item.get('authority','')}|{item.get('instrument','')}|{item.get('date','')}|{item.get('url','')}"
    return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _date_sort_key(date_str: str) -> Tuple[int, str]:
    ds = (date_str or "").strip()
    if ds.lower() == "date unclear":
        return (0, "")
    try:
        if len(ds) == 10:
            dt = datetime.strptime(ds, "%Y-%m-%d")
            return (1, dt.isoformat())
        if len(ds) == 7:
            dt = datetime.strptime(ds, "%Y-%m")
            return (1, dt.isoformat())
    except Exception:
        return (0, "")
    return (0, "")


def _extract_updates(
    llm: OpenAI,
    model_id: str,
    topic_name: str,
    topic_guidance: str,
    today_utc: str,
    window_days: int,
    context: str,
) -> List[dict]:
    user_prompt = (
        f"today={today_utc}\n"
        f"window_days={window_days}\n"
        f"category={topic_name}\n"
        f"Topic guidance: {topic_guidance}\n\n"
        f"Extract ONLY developments within the last window_days relative to today.\n"
        f"If nothing qualifies, return an empty JSON array: [].\n\n"
        f"SOURCES:\n{context}\n"
    )

    try:
        resp = llm.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return []

    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, list):
        return []

    out: List[dict] = []
    for obj in parsed:
        norm = _normalize_item(obj, fallback_category=topic_name)
        if norm:
            out.append(norm)

    out.sort(key=lambda it: _date_sort_key(it.get("date", "")), reverse=True)
    return out


# -------------------------
# Canonical dedupe (URL+instrument first)
# -------------------------
def _canon_text(x: str) -> str:
    x = (x or "").strip().lower()
    x = " ".join(x.split())
    return x


def _canon_url(url: str) -> str:
    url = (url or "").strip()
    if not url.startswith("https://"):
        return ""
    if url.endswith("/"):
        url = url[:-1]
    return url


def _dedupe_key(item: dict) -> str:
    url = _canon_url(item.get("url", ""))
    instrument = _canon_text(item.get("instrument", ""))
    if url:
        return f"url:{url}|inst:{instrument}"
    authority = _canon_text(item.get("authority", ""))
    date = _canon_text(item.get("date", ""))
    return f"aid:{authority}|{instrument}|{date}"


def _dedupe_items_canonical(items: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        k = _dedupe_key(it)
        if not k:
            out.append(it)
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# -------------------------
# Main runner (multi-pass + fulltext fetch + cache)
# -------------------------
def run_watchdog(
    today_utc: str,
    tavily_search_depth: str = "advanced",
    max_results_per_topic: int = 12,
    local_results_per_topic: int = 30,
    preferred_domains: Optional[List[str]] = None,
    window_days: int = 730,
    # quality knobs:
    rerank_top_k: int = 10,
    fetch_fulltext_top_k: int = 6,
    fetch_timeout_sec: int = 20,
    fetch_cache_ttl_hours: int = 72,
    fetch_max_chars_per_url: int = 12000,
    context_max_total_chars: int = 45000,
    polite_delay_sec: float = 0.4,
    progress_callback=None,
    # NEW: feed specific URLs (read regardless of Tavily)
    extra_urls: Optional[List[str]] = None,
) -> dict:
    tav = _tavily_client()
    llm = _groq_client()
    model_id = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    cache = _load_fetch_cache()

    state = load_state()
    existing_items: List[dict] = state.get("items", [])
    if not isinstance(existing_items, list):
        existing_items = []

    existing_items = _dedupe_items_keep_first_by_id(existing_items)
    existing_by_id = {it.get("id"): it for it in existing_items if isinstance(it, dict) and it.get("id")}

    additions: List[dict] = []
    total = len(CATEGORY_TABS_ORDER)

    # If UI passes nothing, fall back to your fixed list
    effective_extra_urls = (extra_urls or []) or list(DEFAULT_EXTRA_URLS)

    for i, topic in enumerate(CATEGORY_TABS_ORDER, start=1):
        guidance = TOPICS[topic]

        if progress_callback:
            try:
                progress_callback(topic, i, total, "search")
            except Exception:
                pass

        mr = int(local_results_per_topic) if topic == LOCAL_CATEGORY else int(max_results_per_topic)

        sources = _search_topic(
            client=tav,
            topic_name=topic,
            search_depth=tavily_search_depth,
            max_results=mr,
            preferred_domains=preferred_domains,
            window_days=window_days,
        )

        # Inject extra URLs ONLY under Local category (so they are not repeated 9 times)
        if topic == LOCAL_CATEGORY and effective_extra_urls:
            sources = _sources_from_urls(effective_extra_urls, topic) + sources

        # Deduplicate by URL
        sources = _dedupe_sources_by_url(sources)

        # Split seed vs non-seed
        seed_sources = [s for s in sources if s.get("_seed")]
        non_seed_sources = [s for s in sources if not s.get("_seed")]

        if progress_callback:
            try:
                progress_callback(topic, i, total, "rerank")
            except Exception:
                pass

        # Rerank only non-seeds; seeds always kept
        selected_non_seed = _rerank_sources(
            llm=llm,
            model_id=model_id,
            topic=topic,
            window_days=window_days,
            sources=non_seed_sources,
            k=int(rerank_top_k),
        )
        selected = _dedupe_sources_by_url(seed_sources + selected_non_seed)

        if progress_callback:
            try:
                progress_callback(topic, i, total, "fetch")
            except Exception:
                pass

        # Always fetch ALL seeds, plus top N non-seeds
        fetch_list = _dedupe_sources_by_url(seed_sources + selected_non_seed[: int(fetch_fulltext_top_k)])

        fulltexts: Dict[str, str] = {}
        for s in fetch_list:
            url = (s.get("url") or "").strip()
            if not url:
                continue
            txt = _fetch_url_text(
                url=url,
                cache=cache,
                cache_ttl_hours=int(fetch_cache_ttl_hours),
                timeout_sec=int(fetch_timeout_sec),
                max_chars=int(fetch_max_chars_per_url),
                polite_delay_sec=float(polite_delay_sec),
            )
            if txt:
                fulltexts[url] = txt

        _save_fetch_cache(cache)

        if progress_callback:
            try:
                progress_callback(topic, i, total, "extract")
            except Exception:
                pass

        context = _build_context_from_sources(
            sources=selected,
            fulltexts=fulltexts,
            max_total_chars=int(context_max_total_chars),
        )

        extracted = _extract_updates(
            llm=llm,
            model_id=model_id,
            topic_name=topic,
            topic_guidance=guidance,
            today_utc=today_utc,
            window_days=window_days,
            context=context,
        )

        for item in extracted:
            item_id = _item_id(item)
            if item_id in existing_by_id:
                existing_by_id[item_id]["last_seen_utc"] = _utc_now_iso()
                continue

            new_item = dict(item)
            new_item["id"] = item_id
            new_item["first_seen_utc"] = _utc_now_iso()
            new_item["last_seen_utc"] = new_item["first_seen_utc"]

            existing_items.insert(0, new_item)
            existing_by_id[item_id] = new_item
            additions.append(new_item)

    existing_items = _dedupe_items_keep_first_by_id(existing_items)
    existing_items.sort(key=lambda it: _date_sort_key((it or {}).get("date", "")), reverse=True)
    existing_items = _dedupe_items_canonical(existing_items)

    state["items"] = existing_items
    save_state(state)
    save_latest_run(additions)

    return {
        "timestamp_utc": _utc_now_iso(),
        "added": additions,
        "all_items": existing_items,
    }
