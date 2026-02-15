# Environmental Specialist Watch Dog

A Streamlit + Tavily + Groq app that continuously monitors maritime environmental regulatory updates, stores deduplicated findings, and shows a simplified UI.

## What improved
- Better source quality: preferred-domain and freshness-aware scoring fallback.
- Better output quality: post-extraction quality gate + strict time-window validation.
- Continuous execution options:
  - **UI continuous mode** (auto-refresh + rerun).
  - **CLI daemon mode** (`python env_watchdog.py --continuous`).
- Simpler UI: fewer controls, cleaner results table, clear new-item counts.

## Prerequisites
- Python 3.11+
- Tavily API key
- Groq API key

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment variables
Required:
- `TAVILY_API_KEY`
- `GROQ_API_KEY`

Optional:
- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)
- `WATCHDOG_EXTRA_URLS` (newline separated https URLs)
- `WINDOW_DAYS` (default 730)
- `MAX_RESULTS_PER_TOPIC` (default 12)
- `LOCAL_RESULTS_PER_TOPIC` (default 30)
- `DATA_DIR` (default `data`)
- `REFRESH_SECONDS` (default 3600; used by CLI defaults)

## Run Streamlit UI
```bash
streamlit run app.py
```
Enable **Continuous mode** in sidebar to keep running at intervals.

## Run in terminal continuously
```bash
python env_watchdog.py --continuous --interval-minutes 60
```

Single run:
```bash
python env_watchdog.py --once
```

## Storage
- `data/state.json` — merged deduplicated items
- `data/latest_run.json` — items added in latest run
- `data/fetch_cache.json` — URL text cache
