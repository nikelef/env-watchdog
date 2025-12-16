import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
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
# Storage (merge-only; dedupe by id)
# -------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
LATEST_RUN_PATH = os.path.join(DATA_DIR, "latest_run.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedupe_items_keep_first(items: List[dict]) -> List[dict]:
    """Remove duplicates by id while preserving order (first occurrence kept)."""
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
        if isinstance(items, list):
            deduped = _dedupe_items_keep_first(items)
            if len(deduped) != len(items):
                state["items"] = deduped
                save_state(state)  # persist dedupe once (no loss of unique findings)
        else:
            state["items"] = []
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
# Retrieval
# -------------------------
def _search_topic(
    client: TavilyClient,
    topic_name: str,
    today_utc: str,
    search_depth: str,
    max_results: int,
    preferred_domains: Optional[List[str]],
    window_days: int,
) -> List[dict]:
    q = (
        f"{topic_name} "
        f"(amendment OR circular OR resolution OR guideline OR enforcement OR delegated act OR regulation) "
        f"(MEPC OR IMO OR flag circular OR class technical news OR EU Commission OR USCG OR AMSA OR CARB) "
        f"last {window_days} days"
    )

    kwargs = {}
    if preferred_domains:
        kwargs["include_domains"] = preferred_domains

    res = client.search(
        query=q,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=False,
        include_images=False,
        **kwargs,
    )

    results = res.get("results", []) if isinstance(res, dict) else []
    for r in results:
        r["_topic"] = topic_name
    return results


def _build_context(sources: List[dict], max_chars: int = 12000) -> str:
    chunks: List[str] = []
    for s in sources:
        title = (s.get("title") or "").strip()
        url = (s.get("url") or "").strip()
        content = (s.get("content") or s.get("snippet") or "").strip()
        topic = (s.get("_topic") or "").strip()

        if not url or not content:
            continue

        block = f"TOPIC: {topic}\nTITLE: {title}\nURL: {url}\nCONTENT:\n{content}\n"
        chunks.append(block)

    text = "\n\n".join(chunks).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n(TRUNCATED)\n"


# -------------------------
# LLM extraction (JSON)
# -------------------------
SYSTEM_PROMPT = (
    "You are an Environmental Specialist Watch Dog for an international ship management company. "
    "You extract only regulatory developments within a time window defined by the user. "
    "Prefer official sources (IMO, flags, regulators, class) and ignore older baseline rules.\n\n"
    "Return STRICT JSON only: a JSON array of objects. No markdown. No extra text.\n"
    "Each object must have keys:\n"
    "category, authority, instrument, date, summary, url\n"
    "date must be 'YYYY-MM-DD' if known; else 'date unclear'.\n"
    "summary must be short and practical (operations/docs/surveys/inspections).\n"
    "url must be https or 'link unavailable'."
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
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


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

    # If it's totally empty, skip
    if authority == "authority unclear" and instrument == "instrument unclear" and summary == "summary unclear" and url == "link unavailable":
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


def _extract_updates_for_topic(
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"LLM request failed for '{topic_name}': {type(e).__name__}: {e}")
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
# Main runner: merge-only + dedupe
# -------------------------
def run_watchdog(
    today_utc: str,
    tavily_search_depth: str = "advanced",
    max_results_per_topic: int = 8,
    local_results_per_topic: int = 20,
    preferred_domains: Optional[List[str]] = None,
    window_days: int = 730,
    progress_callback=None,
) -> dict:
    """
    Returns dict:
      {
        "timestamp_utc": "...",
        "added": [items...],        # new additions in this run
        "all_items": [items...],    # full state after merge
      }
    """
    tav = _tavily_client()
    llm = _groq_client()
    model_id = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    state = load_state()
    existing_items: List[dict] = state.get("items", [])
    existing_items = _dedupe_items_keep_first(existing_items)  # hard guarantee at runtime

    existing_by_id = {it.get("id"): it for it in existing_items if isinstance(it, dict) and it.get("id")}
    additions: List[dict] = []

    total = len(CATEGORY_TABS_ORDER)

    for i, topic in enumerate(CATEGORY_TABS_ORDER, start=1):
        guidance = TOPICS[topic]

        if progress_callback:
            try:
                progress_callback(topic, i, total)
            except Exception:
                pass

        # Flexible: only local category uses a separate max_results
        mr = int(local_results_per_topic) if topic == LOCAL_CATEGORY else int(max_results_per_topic)

        sources = _search_topic(
            client=tav,
            topic_name=topic,
            today_utc=today_utc,
            search_depth=tavily_search_depth,
            max_results=mr,
            preferred_domains=preferred_domains,
            window_days=window_days,
        )

        # Deduplicate sources by URL for context
        seen_urls = set()
        uniq_sources = []
        for s in sources:
            u = (s.get("url") or "").strip()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            uniq_sources.append(s)

        context = _build_context(uniq_sources, max_chars=12000)

        extracted = _extract_updates_for_topic(
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

    # Sort newest first by date; unknown last
    existing_items = _dedupe_items_keep_first(existing_items)
    existing_items.sort(key=lambda it: _date_sort_key((it or {}).get("date", "")), reverse=True)

    state["items"] = existing_items
    save_state(state)
    save_latest_run(additions)

    return {
        "timestamp_utc": _utc_now_iso(),
        "added": additions,
        "all_items": existing_items,
    }
