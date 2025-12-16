import os
from datetime import datetime, timezone

import streamlit as st

from env_watchdog import (
    run_watchdog,
    load_state,
    load_latest_run,
    CATEGORY_TABS_ORDER,
)

st.set_page_config(page_title="Environmental Watch Dog", layout="wide")
st.title("Environmental Watch Dog")
st.caption("2-year window, per-category tables, newest first, merge-only (no deletions).")


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


with st.sidebar:
    st.header("Settings")

    today_override = st.text_input("Today override (YYYY-MM-DD, optional)", value=os.environ.get("TODAY_OVERRIDE", ""))
    window_days = st.number_input("Lookback window (days)", min_value=30, max_value=3650, value=int(os.environ.get("WINDOW_DAYS", "730")))
    max_results_per_topic = st.number_input("Tavily results per topic", min_value=3, max_value=15, value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "8")))
    search_depth = st.selectbox("Tavily search depth", ["basic", "advanced"], index=1)

    include_domains = st.text_area(
        "Preferred domains (optional, comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", ""),
        help="Example: imo.org, europa.eu, amsa.gov.au, uscg.mil, dnv.com, lr.org, bv.com",
    )

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.number_input("Auto-refresh interval (minutes)", min_value=5, max_value=24 * 60, value=120)

    st.divider()
    st.caption("Streamlit Cloud Secrets required:")
    st.code(
        'TAVILY_API_KEY="..."\nGROQ_API_KEY="..."\nGROQ_MODEL="llama-3.1-8b-instant"',
        language="text",
    )

if auto_refresh:
    st.autorefresh(interval=int(refresh_minutes) * 60 * 1000, key="autorefresh")


run_now = st.button("Run now", type="primary")

if run_now or auto_refresh:
    today_utc = _today_utc_iso(today_override)
    result = run_watchdog(
        today_utc=today_utc,
        tavily_search_depth=search_depth,
        max_results_per_topic=int(max_results_per_topic),
        preferred_domains=_parse_domains(include_domains),
        window_days=int(window_days),
    )
    st.success(f"Run completed at {result['timestamp_utc']} UTC. New items added: {len(result['added'])}")

latest_run = load_latest_run()
latest_added_ids = {it.get("id") for it in (latest_run.get("additions") or []) if isinstance(it, dict) and it.get("id")}

state = load_state()
items = state.get("items", [])
if not items:
    st.info("No stored items yet. Click 'Run now'.")
    st.stop()

st.subheader("Results by category (newest first; newest additions highlighted)")

tabs = st.tabs(CATEGORY_TABS_ORDER)

for idx, cat in enumerate(CATEGORY_TABS_ORDER):
    with tabs[idx]:
        cat_items = [it for it in items if isinstance(it, dict) and it.get("category") == cat]

        if not cat_items:
            st.info("No items stored for this category.")
            continue

        # Build table rows
        rows = []
        for it in cat_items:
            rows.append({
                "NEW": "YES" if it.get("id") in latest_added_ids else "",
                "Date": it.get("date", "date unclear"),
                "Authority": it.get("authority", ""),
                "Instrument": it.get("instrument", ""),
                "Practical summary": it.get("summary", ""),
                "URL": it.get("url", ""),
                "First seen (UTC)": it.get("first_seen_utc", ""),
            })

        # Display with simple highlight: NEW column makes it visually obvious.
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        # "Newest developments on top, the last highlighted":
        # - We already sort newest first.
        # - Highlight newest additions via NEW=YES.
        if latest_added_ids:
            st.caption("Rows marked NEW=YES were added in the latest run (highlight for your attention).")
