import os
import html
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
st.caption("2-year window, per-category lists, newest first, merge-only (no deletions).")


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


def _render_category_table(cat: str, cat_items: list, latest_added_ids: set[str]) -> None:
    """
    Render an HTML table so URLs are clickable.
    Highlight the first (latest) row in light blue.
    """
    if not cat_items:
        st.info("No items stored for this category.")
        return

    # Build rows (already sorted newest-first by backend)
    rows = []
    for it in cat_items:
        _id = it.get("id", "")
        rows.append({
            "NEW": "YES" if _id in latest_added_ids else "",
            "Date": it.get("date", "date unclear"),
            "Authority": it.get("authority", ""),
            "Instrument": it.get("instrument", ""),
            "Practical summary": it.get("summary", ""),
            "URL": it.get("url", "link unavailable"),
            "First seen (UTC)": it.get("first_seen_utc", ""),
        })

    # HTML table with hover and clickable links
    # Highlight latest row (row 0) light blue.
    css = """
    <style>
      .wd-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
      .wd-table th, .wd-table td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
      .wd-table th { background: #f4f6f8; text-align: left; }
      .wd-latest { background: #d9ecff; } /* light blue */
      .wd-new { font-weight: 600; }
      .wd-url a { text-decoration: none; }
      .wd-url a:hover { text-decoration: underline; }
    </style>
    """

    header_cols = ["NEW", "Date", "Authority", "Instrument", "Practical summary", "URL", "First seen (UTC)"]

    html_rows = []
    for i, r in enumerate(rows):
        tr_class = "wd-latest" if i == 0 else ""
        new_class = "wd-new" if r["NEW"] == "YES" else ""

        url_val = r["URL"]
        if url_val.startswith("https://"):
            url_html = f'<span class="wd-url"><a href="{html.escape(url_val)}" target="_blank" rel="noopener noreferrer">{html.escape(url_val)}</a></span>'
        else:
            url_html = html.escape(url_val)

        html_rows.append(
            f"""
            <tr class="{tr_class}">
              <td class="{new_class}">{html.escape(r["NEW"])}</td>
              <td>{html.escape(r["Date"])}</td>
              <td>{html.escape(r["Authority"])}</td>
              <td>{html.escape(r["Instrument"])}</td>
              <td>{html.escape(r["Practical summary"])}</td>
              <td>{url_html}</td>
              <td>{html.escape(r["First seen (UTC)"])}</td>
            </tr>
            """
        )

    table_html = f"""
    {css}
    <table class="wd-table">
      <thead>
        <tr>
          {''.join([f"<th>{html.escape(c)}</th>" for c in header_cols])}
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("Top row (light blue) is the latest entry in this category. NEW=YES indicates added in the latest run.")


with st.sidebar:
    st.header("Settings")

    today_override = st.text_input("Today override (YYYY-MM-DD, optional)", value=os.environ.get("TODAY_OVERRIDE", ""))
    window_days = st.number_input(
        "Lookback window (days)", min_value=30, max_value=3650, value=int(os.environ.get("WINDOW_DAYS", "730"))
    )
    max_results_per_topic = st.number_input(
        "Tavily results per topic", min_value=3, max_value=15, value=int(os.environ.get("MAX_RESULTS_PER_TOPIC", "8"))
    )
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
    st.code('TAVILY_API_KEY="..."\nGROQ_API_KEY="..."\nGROQ_MODEL="llama-3.1-8b-instant"', language="text")

if auto_refresh:
    st.autorefresh(interval=int(refresh_minutes) * 60 * 1000, key="autorefresh")

run_now = st.button("Run now", type="primary")

# ---- Progress indicator (only) ----
status_box = st.status("Idle", expanded=False)
progress_bar = st.progress(0)

def _progress_cb(topic: str, idx: int, total: int) -> None:
    # idx is 1..total
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
        preferred_domains=_parse_domains(include_domains),
        window_days=int(window_days),
        progress_callback=_progress_cb,   # <-- progress only
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

st.subheader("Results (single-page scroll, newest-first per category)")

for cat in CATEGORY_TABS_ORDER:
    st.markdown(f"### {cat}")
    cat_items = [it for it in items if isinstance(it, dict) and it.get("category") == cat]
    _render_category_table(cat, cat_items, latest_added_ids)
    st.markdown("---")
