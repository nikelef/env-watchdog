# app.py
import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import streamlit as st
from openai import OpenAI

from env_watchdog import (
    run_watchdog,
    load_state,
    load_latest_run,
    CATEGORY_TABS_ORDER,
    LOCAL_CATEGORY,
    DEFAULT_EXTRA_URLS,   # NEW: used to prefill the UI
)

ALERT_RECIPIENT = "neleftheriou@tms-dry.com"

st.set_page_config(page_title="Environmental Watch Dog", layout="wide")
st.title("Environmental Watch Dog")
st.caption("Multi-pass search + source rerank + full-text fetch (cached). Collapsible categories. Flag rows to draft emails.")


# ---- CSS: wrap text inside Streamlit data_editor/dataframe cells ----
st.markdown(
    """
    <style>
    /* Wrap in dataframe/editor cells */
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataEditor"] div[role="gridcell"] {
        white-space: normal !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _parse_domains(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _today_utc_iso(today_override: str) -> str:
    today_override = (today_override or "").strip()
    if today_override:
        return today_override
    return datetime.now(timezone.utc).date().isoformat()


def _groq_client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)


def _email_paragraph_300w(item: dict) -> str:
    model_id = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    llm = _groq_client()

    authority = item.get("authority", "authority unclear")
    instrument = item.get("instrument", "instrument unclear")
    date = item.get("date", "date unclear")
    url = item.get("url", "link unavailable")
    short_summary = (item.get("summary") or "").replace("\n", " ").strip()

    prompt = (
        "Write ONE paragraph (no bullets) of about 180-220 words suitable for an internal company email. "
        "Summarize the regulatory development below in practical shipping terms. "
        "Focus on: what changed, effective/enforcement timing if known, operational impact, "
        "documentation/survey/PSC focus. Do not invent facts. If uncertain, state that confirmation "
        "from official sources is needed.\n\n"
        f"Authority: {authority}\n"
        f"Instrument: {instrument}\n"
        f"Date: {date}\n"
        f"Known short summary: {short_summary}\n"
        f"Link: {url}\n"
    )

    resp = llm.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You write concise, factual regulatory email summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    out = (resp.choices[0].message.content or "").strip()
    out = " ".join(out.splitlines()).strip()  # enforce one paragraph
    return out


def build_email_draft(recipient: str, item: dict, paragraph: str, today_utc: str) -> tuple[str, str]:
    subject = f"Environmental regulatory update - {item.get('authority','')} {item.get('instrument','')} - {today_utc}".strip()
    url = item.get("url") or "link unavailable"
    date = item.get("date") or "date unclear"
    authority = item.get("authority") or "authority unclear"
    instrument = item.get("instrument") or "instrument unclear"

    body = (
        f"To: {recipient}\n"
        f"Subject: {subject}\n\n"
        f"Dear all,\n\n"
        f"{paragraph}\n\n"
        f"Reference: {authority} - {instrument} - {date}\n"
        f"Link: {url}\n\n"
        f"Best regards,\n"
        f"Villager"
    )
    return subject, body


def _category_df(cat_items: list, latest_added_ids: set[str]) -> pd.DataFrame:
    seen = set()
    deduped = []
    for it in cat_items:
        _id = it.get("id")
        if not _id or _id in seen:
            continue
        seen.add(_id)
        deduped.append(it)

    rows = []
    for idx, it in enumerate(deduped):
        _id = it.get("id", "")
        url = it.get("url", "link unavailable")
        if not isinstance(url, str):
            url = "link unavailable"
        url = url if url.startswith("https://") else "link unavailable"

        rows.append(
            {
                "Flag": False,
                "Latest": "LATEST" if idx == 0 else "",
                "NEW": "YES" if _id in latest_added_ids else "",
                "Date": it.get("date", "date unclear"),
                "Authority": it.get("authority", ""),
                "Instrument": it.get("instrument", ""),
                "Practical summary": it.get("summary", ""),
                "URL": url,
                "_id": _id,
            }
        )
    return pd.DataFrame(rows)


def _parse_urls_text(raw: str) -> List[str]:
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


with st.sidebar:
    st.header("Settings")

    today_override = st.text_input("Today override (YYYY-MM-DD, optional)", value=os.environ.get("TODAY_OVERRIDE", ""))
    window_days = st.number_input("Lookback window (days)", min_value=30, max_value=3650, value=int(os.environ.get("WINDOW_DAYS", "730")))
    search_depth = st.selectbox("Tavily search depth", ["basic", "advanced"], index=1)

    max_results_per_topic = st.number_input("Search results per topic (non-Local)", min_value=5, max_value=40, value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "12")))
    local_results_per_topic = st.number_input("Search results per topic (Local only)", min_value=10, max_value=80, value=int(os.environ.get("LOCAL_RESULTS_PER_TOPIC", "30")))

    rerank_top_k = st.number_input("Rerank: candidate sources kept (per topic)", min_value=5, max_value=30, value=int(os.environ.get("RERANK_TOP_K", "10")))
    fetch_fulltext_top_k = st.number_input("Fetch full-text for top N sources (per topic)", min_value=0, max_value=20, value=int(os.environ.get("FETCH_FULLTEXT_TOP_K", "6")))
    fetch_timeout_sec = st.number_input("Fetch timeout per URL (sec)", min_value=5, max_value=60, value=int(os.environ.get("FETCH_TIMEOUT_SEC", "20")))
    fetch_cache_ttl_hours = st.number_input("Fetch cache TTL (hours)", min_value=1, max_value=720, value=int(os.environ.get("FETCH_CACHE_TTL_HOURS", "72")))
    polite_delay_sec = st.number_input("Polite delay between fetches (sec)", min_value=0.0, max_value=3.0, value=float(os.environ.get("POLITE_DELAY_SEC", "0.4")))

    include_domains = st.text_area(
        "Preferred domains (optional, comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", ""),
        help="Example: imo.org, europa.eu, amsa.gov.au, uscg.mil, epa.gov, carb.ca.gov, dnv.com, lr.org",
    )

    st.divider()
    st.subheader("Always-read URLs (in addition to search)")
    default_urls_text = os.environ.get("WATCHDOG_EXTRA_URLS", "\n".join(DEFAULT_EXTRA_URLS))
    extra_urls_text = st.text_area(
        "Paste URLs (one per line)",
        value=default_urls_text,
        height=220,
        help="These URLs are always fetched/read and injected under the Local category, even if Tavily fails.",
    )

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.number_input("Auto-refresh interval (minutes)", min_value=5, max_value=24 * 60, value=120)

    st.divider()
    st.caption("Streamlit Cloud Secrets required:")
    st.code('TAVILY_API_KEY="..."\nGROQ_API_KEY="..."\nGROQ_MODEL="llama-3.1-8b-instant"', language="text")
    st.caption("Optional (to persist URLs via Secrets):")
    st.code('WATCHDOG_EXTRA_URLS="https://...\\nhttps://..."', language="text")


if auto_refresh:
    st.autorefresh(interval=int(refresh_minutes) * 60 * 1000, key="autorefresh")

run_now = st.button("Run now", type="primary")

status_box = st.status("Idle", expanded=False)
progress_bar = st.progress(0)

PHASE_LABELS = {
    "search": "Searching",
    "rerank": "Reranking sources",
    "fetch": "Fetching full text",
    "extract": "Extracting updates",
}


def _progress_cb(topic: str, idx: int, total: int, phase: str) -> None:
    phase_txt = PHASE_LABELS.get(phase, phase)
    status_box.update(label=f"{phase_txt}: {idx}/{total} — {topic}", state="running", expanded=False)
    pct = int((idx / max(total, 1)) * 100)
    progress_bar.progress(pct)


today_utc = _today_utc_iso(today_override)

if run_now or auto_refresh:
    status_box.update(label="Starting run…", state="running", expanded=False)
    progress_bar.progress(0)

    extra_urls = _parse_urls_text(extra_urls_text)

    result = run_watchdog(
        today_utc=today_utc,
        tavily_search_depth=search_depth,
        max_results_per_topic=int(max_results_per_topic),
        local_results_per_topic=int(local_results_per_topic),
        preferred_domains=_parse_domains(include_domains),
        window_days=int(window_days),
        rerank_top_k=int(rerank_top_k),
        fetch_fulltext_top_k=int(fetch_fulltext_top_k),
        fetch_timeout_sec=int(fetch_timeout_sec),
        fetch_cache_ttl_hours=int(fetch_cache_ttl_hours),
        polite_delay_sec=float(polite_delay_sec),
        progress_callback=_progress_cb,
        extra_urls=extra_urls,  # NEW
    )

    status_box.update(label=f"Run completed at {result['timestamp_utc']} UTC", state="complete", expanded=False)
    progress_bar.progress(100)
    st.success(f"Run completed. New items added: {len(result['added'])}")

latest_run = load_latest_run()
latest_added_ids = {
    it.get("id")
    for it in (latest_run.get("additions") or [])
    if isinstance(it, dict) and it.get("id")
}

state = load_state()
items = state.get("items", [])
if not items:
    st.info("No stored items yet. Click 'Run now'.")
    st.stop()

st.subheader("Flag items and generate email drafts")
gen_drafts = st.button("Generate email draft(s) for flagged", type="secondary")

if "drafts" not in st.session_state:
    st.session_state["drafts"] = []

st.caption("Tick Flag next to any entry, then generate email drafts (1 paragraph, ~300 words).")

flagged_requests: List[dict] = []

st.subheader("Results (click category to expand/collapse)")

for cat in CATEGORY_TABS_ORDER:
    cat_items = [it for it in items if isinstance(it, dict) and it.get("category") == cat]
    df = _category_df(cat_items, latest_added_ids)

    with st.expander(f"{cat} ({len(df)})", expanded=False):
        if df.empty:
            st.info("No items stored for this category.")
            continue

        edited = st.data_editor(
            df.drop(columns=[], errors="ignore"),
            hide_index=True,
            use_container_width=True,
            key=f"editor_{cat}",
            disabled=["Latest", "NEW", "Date", "Authority", "Instrument", "Practical summary", "URL", "_id"],
            column_config={
                "Flag": st.column_config.CheckboxColumn("Flag", help="Flag to generate email draft"),
                "Latest": st.column_config.TextColumn("Latest", width="small"),
                "NEW": st.column_config.TextColumn("NEW", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Authority": st.column_config.TextColumn("Authority", width="medium"),
                "Instrument": st.column_config.TextColumn("Instrument", width="medium"),
                "Practical summary": st.column_config.TextColumn("Practical summary", width="large"),
                "URL": st.column_config.LinkColumn("URL", display_text="Open link"),
                "_id": st.column_config.TextColumn("_id", width="small"),
            },
        )

        try:
            edited_df = pd.DataFrame(edited)
        except Exception:
            edited_df = df.copy()

        if "Flag" in edited_df.columns and "_id" in edited_df.columns:
            flagged_ids = edited_df.loc[edited_df["Flag"] == True, "_id"].tolist()
            for fid in flagged_ids:
                flagged_requests.append({"category": cat, "id": fid})

# Generate drafts
if gen_drafts:
    by_id = {it.get("id"): it for it in items if isinstance(it, dict) and it.get("id")}
    drafts_out = []

    for req in flagged_requests:
        it = by_id.get(req["id"])
        if not it:
            continue

        try:
            paragraph = _email_paragraph_300w(it)
        except Exception as e:
            paragraph = f"Could not generate summary due to error: {type(e).__name__}: {e}"

        _, draft = build_email_draft(ALERT_RECIPIENT, it, paragraph, today_utc)
        drafts_out.append(
            {
                "category": it.get("category", ""),
                "authority": it.get("authority", ""),
                "instrument": it.get("instrument", ""),
                "date": it.get("date", ""),
                "url": it.get("url", ""),
                "draft": draft,
            }
        )

    st.session_state["drafts"] = drafts_out

# Show drafts
if st.session_state.get("drafts"):
    st.markdown("### Email draft(s) (copy/paste)")
    for i, d in enumerate(st.session_state["drafts"], start=1):
        title = f"{i}) {d.get('category','')} - {d.get('authority','')} - {d.get('instrument','')} - {d.get('date','')}"
        st.markdown(f"**{title}**")
        st.text_area(
            label="",
            value=d.get("draft", ""),
            height=280,
            key=f"draft_{i}",
        )
