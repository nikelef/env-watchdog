# app.py
import os
from datetime import datetime, timezone
from typing import List, Optional, Set

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from env_watchdog import (
    CATEGORY_TABS_ORDER,
    DEFAULT_EXTRA_URLS,
    get_missing_credentials,
    load_latest_run,
    load_state,
    run_watchdog,
)

st.set_page_config(page_title="Env Watchdog", layout="wide")
st.title("Env Watchdog")
st.caption("Focused maritime environmental regulatory monitoring (free-tier mode).")


def _today_utc_iso() -> str:
    override = os.environ.get("TODAY_OVERRIDE", "").strip()
    return override or datetime.now(timezone.utc).date().isoformat()


def _parse_domains(raw: str) -> Optional[List[str]]:
    vals = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return vals or None


def _parse_urls_text(raw: str) -> List[str]:
    vals: List[str] = []
    seen = set()
    for line in (raw or "").splitlines():
        item = line.strip()
        if item and item.startswith("https://") and item not in seen:
            seen.add(item)
            vals.append(item)
    return vals


def _to_df(items: list, latest_ids: Set[str]) -> pd.DataFrame:
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "NEW": "YES" if item.get("id") in latest_ids else "",
                "Date": item.get("date", "date unclear"),
                "Authority": item.get("authority", ""),
                "Instrument": item.get("instrument", ""),
                "Summary": (item.get("summary", "") or "").replace("\n", " | "),
                "URL": item.get("url", "link unavailable"),
            }
        )
    return pd.DataFrame(rows)


missing = get_missing_credentials()
if missing:
    st.error(
        "Missing required secrets: "
        + ", ".join(missing)
        + "\n\nAdd them in Streamlit Community Cloud → App settings → Secrets."
    )
    st.stop()


with st.sidebar:
    st.subheader("Run settings (cost control)")

    auto_refresh = st.toggle("Continuous mode", value=False)
    if auto_refresh:
        st.warning("Continuous mode will repeatedly call the APIs and may hit free-tier limits.", icon="⚠️")

    refresh_minutes = st.slider("Run interval (minutes)", min_value=10, max_value=720, value=180, step=10)
    window_days = st.slider("Lookback window (days)", min_value=30, max_value=1825, value=365, step=30)
    max_results = st.slider("Search results per topic", min_value=5, max_value=40, value=10)
    local_results = st.slider("Search results for local/regional topic", min_value=10, max_value=80, value=25)

    search_depth = st.selectbox("Search depth", ["basic", "advanced"], index=0)

    preferred_domains = st.text_input(
        "Preferred domains (comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", "imo.org,europa.eu,amsa.gov.au,uscg.mil"),
    )

    extra_urls_text = st.text_area(
        "Always-read URLs (only injected into Local/Regional category)",
        value=os.environ.get("WATCHDOG_EXTRA_URLS", "\n".join(DEFAULT_EXTRA_URLS)),
        height=180,
    )

    if auto_refresh:
        refresh_ms = int(refresh_minutes) * 60 * 1000
        components.html(
            f"<script>setTimeout(function(){{window.location.reload();}}, {refresh_ms});</script>",
            height=0,
        )

    run_now = st.button("Run now", type="primary")


if run_now or auto_refresh:
    with st.status("Running watchdog...", expanded=False) as status:
        result = run_watchdog(
            today_utc=_today_utc_iso(),
            tavily_search_depth=search_depth,
            window_days=int(window_days),
            max_results_per_topic=int(max_results),
            local_results_per_topic=int(local_results),
            preferred_domains=_parse_domains(preferred_domains),
            extra_urls=_parse_urls_text(extra_urls_text),
        )
        status.update(label=f"Completed. New updates: {len(result.get('added') or [])}", state="complete")


latest_run = load_latest_run()
latest_added_ids = {
    it.get("id")
    for it in (latest_run.get("additions") or [])
    if isinstance(it, dict) and it.get("id")
}

state = load_state()
items = [it for it in (state.get("items") or []) if isinstance(it, dict)]

if not items:
    st.info("No results yet. Click **Run now**.")
    st.stop()

st.metric("Total stored updates", len(items))
st.metric("New in last run", len(latest_added_ids))

st.subheader("Regulatory updates")
for category in CATEGORY_TABS_ORDER:
    cat_items = [it for it in items if it.get("category") == category]
    with st.expander(f"{category} ({len(cat_items)})", expanded=False):
        df = _to_df(cat_items, latest_added_ids)
        if df.empty:
            st.info("No updates in this category.")
        else:
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={"URL": st.column_config.LinkColumn("URL", display_text="open")},
            )
