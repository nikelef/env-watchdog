import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from env_watchdog import (
    CATEGORY_TABS_ORDER,
    DEFAULT_EXTRA_URLS,
    load_latest_run,
    load_state,
    run_watchdog,
)

st.set_page_config(page_title="Env Watchdog", layout="wide")
st.title("Env Watchdog")
st.caption("Focused maritime environmental regulatory monitoring.")


def _required_keys_present() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY", "").strip()) and bool(os.environ.get("GROQ_API_KEY", "").strip())


def _today_utc_iso() -> str:
    override = os.environ.get("TODAY_OVERRIDE", "").strip()
    return override or datetime.now(timezone.utc).date().isoformat()


def _parse_domains(raw: str):
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


def _to_df(items: list, latest_ids: set[str]) -> pd.DataFrame:
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
                "Summary": item.get("summary", "").replace("\n", " | "),
                "URL": item.get("url", "link unavailable"),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("Run settings")
    auto_refresh = st.toggle("Continuous mode", value=False)
    refresh_minutes = st.slider("Run interval (minutes)", min_value=5, max_value=720, value=120, step=5)
    window_days = st.slider("Lookback window (days)", min_value=30, max_value=1825, value=int(os.environ.get("WINDOW_DAYS", "730")), step=30)
    max_results = st.slider("Search results per topic", min_value=5, max_value=40, value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "12")))
    local_results = st.slider("Search results for local/regional topic", min_value=10, max_value=80, value=int(os.environ.get("LOCAL_RESULTS_PER_TOPIC", "30")))
    search_depth = st.selectbox("Search depth", ["advanced", "basic"], index=0)
    preferred_domains = st.text_input("Preferred domains (comma-separated)", value=os.environ.get("PREFERRED_DOMAINS", "imo.org,europa.eu,amsa.gov.au,uscg.mil"))
    extra_urls_text = st.text_area("Always-read URLs", value=os.environ.get("WATCHDOG_EXTRA_URLS", "\n".join(DEFAULT_EXTRA_URLS)), height=180)

if auto_refresh:
    refresh_ms = refresh_minutes * 60 * 1000
    components.html(
        f"""<script>setTimeout(function() {{ window.parent.location.reload(); }}, {refresh_ms});</script>""",
        height=0,
    )

run_now = st.button("Run now", type="primary")

if run_now or auto_refresh:
    if not _required_keys_present():
        st.error("Missing required secrets: set TAVILY_API_KEY and GROQ_API_KEY before running.")
    else:
        with st.status("Running watchdog...", expanded=False) as status:
            try:
                result = run_watchdog(
                    today_utc=_today_utc_iso(),
                    tavily_search_depth=search_depth,
                    window_days=window_days,
                    max_results_per_topic=max_results,
                    local_results_per_topic=local_results,
                    preferred_domains=_parse_domains(preferred_domains),
                    extra_urls=_parse_urls_text(extra_urls_text),
                )
                status.update(label=f"Completed. New updates: {len(result.get('added') or [])}", state="complete")
            except Exception as exc:
                status.update(label="Run failed", state="error", expanded=True)
                st.exception(exc)

latest_run = load_latest_run()
latest_added_ids = {it.get("id") for it in (latest_run.get("additions") or []) if isinstance(it, dict)}
state = load_state()
items = [it for it in (state.get("items") or []) if isinstance(it, dict)]

if not items:
    st.info("No results yet. Click Run now.")
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
