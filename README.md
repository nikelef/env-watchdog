Environmental Specialist Watch Dog (Streamlit + Ollama + Tavily)

What it does
- Pulls recent web sources per topic (Tavily Search)
- Summarizes with a local Ollama model
- Enforces strict output format:
  - Either EXACT: NO_UPDATES
  - Or up to 8 bullets: "- Authority - Instrument - Date - short practical summary - URL"
- Posts results by saving:
  - data/latest.txt
  - data/history.jsonl

Prerequisites
1) Install Ollama: https://github.com/ollama/ollama
2) Pull the default model:
   ollama pull qwen2.5:7b-instruct
   (Optional alternative)
   ollama pull llama3.1:8b

3) Create a Tavily API key: https://docs.tavily.com/

Setup
- Create a virtual environment, install dependencies:
  pip install -r requirements.txt

Environment variables
- TAVILY_API_KEY=...
- OLLAMA_MODEL=qwen2.5:7b-instruct
- TODAY_OVERRIDE=YYYY-MM-DD   (optional)
- MAX_RESULTS_PER_TOPIC=5     (optional)
- TAVILY_SEARCH_DEPTH=advanced (optional)
- PREFERRED_DOMAINS=imo.org,europa.eu,amsa.gov.au,uscg.mil,dnv.com (optional)
- DATA_DIR=data (optional)
- REFRESH_SECONDS=3600 (optional)

Run
- streamlit run app.py

Notes
- The local model does not browse the web itself; Tavily provides retrieval.
- If the model output violates format rules, the app will fall back to NO_UPDATES.
