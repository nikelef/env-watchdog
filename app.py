import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from env_watchdog import (
    run_watchdog,
    load_state,
    load_latest_run,
    CATEGORY_TABS_ORDER,
    LOCAL_CATEGORY,
)

st.set_page_config(page_title="Environmental Watch Dog", layout="wide")
st.title("Environmental Watch Dog")
st.caption("2-year window, per-category, collapsible, newest first, merge-only (no deletions).")


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


def _render_category_df(cat_items: list, latest_added_ids: set[str]) -> None:
    if not cat_items:
        st.info("No items stored for this category.")
        return

    # Display-level dedupe by id (defensive)
    seen = set()
    deduped = []
    for it in cat_items:
        _id = it.get("id")
        if not _id or _id in seen:
            continue
        seen.add(_id)
        deduped.append(it)

    rows = []
    for it in deduped:
        _id = it.get("id", "")
        url = it.get("url", "link unavailable")
        if not isinstance(url, str):
            url = "link unavailable"

        rows.append(
            {
                "NEW": "YES" if _id in latest_added_ids else "",
                "Date": it.get("date", "date unclear"),
                "Authority": it.get("authority", ""),
                "Instrument": it.get("instrument", ""),
                "Practical summary": it.get("summary", ""),
                "URL": url if url.startswith("https://") else "link unavailable",
                "First seen (UTC)": it.get("first_seen_utc", ""),
            }
        )

    df = pd.DataFrame(rows)

    # Highlight the latest row (row 0) light blue
    def _style_latest(row_idx: int):
        if row_idx == 0:
            return ["background-color: #d9ecff"] * len(df.columns)
        return [""] * len(df.columns)

    styled = df.style.apply(lambda _row: _style_latest(_row.name), axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn(
                "URL",
                help="Click to open",
                validate="^https://.*|link unavailable$",
                display_text="Open link",
            ),
        },
    )
    st.caption("Top row (light blue) is the latest entry in this category. NEW=YES indicates added in the latest run.")


with st.sidebar:
    st.header("Settings")

    today_override = st.text_input("Today override (YYYY-MM-DD, optional)", value=os.environ.get("TODAY_OVERRIDE", ""))
    window_days = st.number_input(
        "Lookback window (days)",
        min_value=30,
        max_value=3650,
        value=int(os.environ.get("WINDOW_DAYS", "730")),
    )

    search_depth = st.selectbox("Tavily search depth", ["basic", "advanced"], index=1)

    max_results_per_topic = st.number_input(
        "Tavily results per topic (all categories except Local/Regional)",
        min_value=3,
        max_value=15,
        value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "8")),
    )

    local_results_per_topic = st.number_input(
        "Tavily results per topic (Local/Regional only)",
        min_value=5,
        max_value=40,
        value=int(os.environ.get("LOCAL_RESULTS_PER_TOPIC", "20")),
        help=f"Applies only to: {LOCAL_CATEGORY}",
    )

    include_domains = st.text_area(
        "Preferred domains (optional, comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", ""),
        help="Example: imo.org, europa.eu, amsa.gov.au, uscg.mil, epa.gov, carbc.ca.gov, dnv.com, lr.org",
    )

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.number_input("Auto-refresh interval (minutes)", min_value=5, max_value=24 * 60, value=120)

    st.divider()
    st.caption("Streamlit Cloud Secrets required:")
    st.code('TAVILY_API_KEY="..."\nGROQ_API_KEY="..."\nGROQ_MODEL="llama-3.1-8b-instant"', language="text")

if auto_refresh:
    st.autorefresh(interval=int(refresh_minutes) * 60 * 1000, key="autorefresh")

run_now = st.button("Run now", type="primary")

# Progress indicator
status_box = st.status("Idle", expanded=False)
progress_bar = st.progress(0)

def _progress_cb(topic: str, idx: int, total: int) -> None:
    status_box.update(label=f"Running: {idx}/{total} — {topic}", state="running", expanded=False)
    pct = int((idx / max(total, 1)) * 100)
    progress_bar.progress(pct)

if run_now or auto_refresh:
    today_utc = _today_utc_iso(today_override)
    status_box.update(label="Starting run…", state="running", expanded=False)
    progress_bar.progress(0)

    result = run_watchdog(
        today_utc=today_utc,
        tavily_search_depth=search_depth,
        max_results_per_topic=int(max_results_per_topic),
        local_results_per_topic=int(local_results_per_topic),
        preferred_domains=_parse_domains(include_domains),
        window_days=int(window_days),
        progress_callback=_progress_cb,
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

st.subheader("Results (click category to expand/collapse)")

for cat in CATEGORY_TABS_ORDER:
    cat_items = [it for it in items if isinstance(it, dict) and it.get("category") == cat]
    with st.expander(f"{cat} ({len(cat_items)})", expanded=False):
        _render_category_df(cat_items, latest_added_ids)
