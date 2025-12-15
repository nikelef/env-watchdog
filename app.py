import os
import time
from datetime import datetime, timezone
import streamlit as st

from watchdog import run_watchdog, load_latest, load_history_tail, save_result

st.set_page_config(page_title="Environmental Specialist Watch Dog", layout="wide")

st.title("Environmental Specialist Watch Dog")
st.caption("Local LLM (Ollama) + Web retrieval (Tavily) + strict output contract")

with st.sidebar:
    st.header("Runtime")
    model = st.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    today_override = st.text_input("Today override (YYYY-MM-DD, optional)", value=os.environ.get("TODAY_OVERRIDE", ""))
    max_results_per_topic = st.number_input("Search results per topic", min_value=2, max_value=10, value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "5")))
    search_depth = st.selectbox("Tavily search_depth", ["basic", "advanced"], index=1 if os.environ.get("TAVILY_SEARCH_DEPTH", "advanced") == "advanced" else 0)
    include_domains = st.text_area(
        "Preferred domains (optional, comma-separated)",
        value=os.environ.get("PREFERRED_DOMAINS", ""),
        help="Optional: add domains to bias results (e.g., imo.org, europa.eu, uscg.mil, amsa.gov.au, dnv.com)"
    )
    st.divider()
    auto_refresh = st.checkbox("Auto-run periodically", value=False)
    refresh_seconds = st.number_input("Auto-run interval (seconds)", min_value=60, max_value=24*3600, value=int(os.environ.get("REFRESH_SECONDS", "3600")))

def _parse_domains(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Latest posted result")
    latest = load_latest()
    if latest:
        st.code(latest, language="text")
    else:
        st.info("No result posted yet. Click Run now.")

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

def _get_today_utc_iso():
    if today_override.strip():
        return today_override.strip()
    return datetime.now(timezone.utc).date().isoformat()

def _do_run():
    today_utc = _get_today_utc_iso()
    output = run_watchdog(
        today_utc=today_utc,
        ollama_model=model.strip(),
        tavily_search_depth=search_depth,
        max_results_per_topic=int(max_results_per_topic),
        preferred_domains=_parse_domains(include_domains),
    )
    save_result(output)
    st.success("Posted (saved) new result.")
    st.code(output, language="text")

if run_now:
    _do_run()

if auto_refresh:
    # simple foreground loop; run only while page is open
    st.warning("Auto-run is active while this page stays open.")
    placeholder = st.empty()
    while True:
        with placeholder.container():
            st.write(f"Auto-run at UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
            try:
                _do_run()
            except Exception as e:
                st.error(f"Auto-run failed: {e}")
        time.sleep(int(refresh_seconds))
