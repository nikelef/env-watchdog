import os
from datetime import datetime, timezone

import streamlit as st

from env_watchdog import run_watchdog, load_latest, load_history_tail, save_result


st.set_page_config(page_title="Environmental Specialist Watch Dog", layout="wide")
st.title("Environmental Specialist Watch Dog")
st.caption("Online mode: Tavily retrieval + Groq (OpenAI-compatible) summarization + strict output contract")


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
    max_results_per_topic = st.number_input(
        "Tavily results per topic",
        min_value=2,
        max_value=10,
        value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "5")),
    )
    search_depth = st.selectbox(
        "Tavily search depth",
        ["basic", "advanced"],
        index=1 if os.environ.get("TAVILY_SEARCH_DEPTH", "advanced") == "advanced" else 0,
    )

    include_domains = st.text_area(
        "Preferred domains (optional, comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", ""),
        help="Example: imo.org, europa.eu, amsa.gov.au, uscg.mil, dnv.com",
    )

    st.divider()
    auto_refresh = st.checkbox("Auto-run periodically", value=False)
    refresh_minutes = st.number_input("Auto-run interval (minutes)", min_value=5, max_value=24 * 60, value=60)

    st.divider()
    st.caption("Secrets required in Streamlit Cloud:")
    st.code('TAVILY_API_KEY="..."\nGROQ_API_KEY="..."\nGROQ_MODEL="llama-3.1-8b-instant"', language="text")


# Auto-refresh without infinite loops (Streamlit Cloud safe)
if auto_refresh:
    st.autorefresh(interval=int(refresh_minutes) * 60 * 1000, key="watchdog_autorefresh")


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Latest posted result")
    latest = load_latest()
    if latest:
        st.code(latest, language="text")
    else:
        st.info("No result posted yet.")

with col2:
    st.subheader("History (most recent first)")
    hist = load_history_tail(limit=50)
    if hist:
        for item in hist:
            st.markdown(f"**{item['timestamp_utc']}**")
            st.code(item["output"], language="text")
            st.markdown("---")
    else:
        st.info("No history yet.")

st.divider()

run_now = st.button("Run now", type="primary")

# If autorefresh is enabled, run automatically on each refresh.
should_run = run_now or auto_refresh

if should_run:
    today_utc = _today_utc_iso(today_override)

    output = run_watchdog(
        today_utc=today_utc,
        tavily_search_depth=search_depth,
        max_results_per_topic=int(max_results_per_topic),
        preferred_domains=_parse_domains(include_domains),
    )

    save_result(output)
    st.success("Posted (saved) new result.")
    st.code(output, language="text")
