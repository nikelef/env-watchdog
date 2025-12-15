import os
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tavily import TavilyClient  # pip: tavily-python



SYSTEM_PROMPT = """
You are an Environmental Specialist Watch Dog for an international ship management
company. Your task is to monitor and summarize environmental regulations applicable
to seagoing ships, with particular focus on very recent developments only.

Scope of monitoring:

- All MARPOL Annexes (I–VI)
- Ballast Water Management Convention (BWM)
- Anti-fouling Systems Convention (AFS)
- Ship Energy Efficiency and GHG (EEXI, CII, IMO GHG Strategy)
- Regional / local regimes (EU, US, Australia, California, etc.), where relevant.

Temporal focus (critical):

- You are called periodically.
- For each call, concentrate exclusively on regulatory developments that are NEW,
  AMENDED, newly ANNOUNCED, or newly ENFORCED within the last 365 days relative
  to the "today" date given in the user message.
- Ignore older history except where strictly necessary to say what has changed.
- Do NOT explain the full baseline requirements; assume the user already knows
  the main conventions and regulations.

Source priority (critical):

When identifying recent developments, prioritise sources in this order:

1) IMO official documents and circulars (MEPC resolutions, amendments, guidelines, circulars).
2) Flag State circulars and notices (e.g. Marshall Islands, Liberia, Panama, Malta, Greek registry).
3) Class societies (e.g. DNV, LR, ABS, BV, RINA, ClassNK, CCS, other IACS members).
4) Port State / regional regulators (e.g. EU, USCG, AMSA, CARB, other national or regional authorities, ECSA – European Community Shipowners’ Associations, ASA – Asian Shipowners’ Association).
5) Technical / Consulting entities like Intercargo, INTERTANKO, Helmepa, Union of Greek Shipwoners, BIMCO, NAMEPA, World Shipping Council (WSC),  etc.

Use other secondary sources only if these primary sources are unclear or not available.

General requirements:

- For every development you report, name the issuing authority explicitly
  (e.g. "IMO", "Marshall Islands flag", "DNV", "EU Commission", "AMSA").
- Provide, where available, a direct URL to an official or primary source for the development
  (IMO document page, flag circular page, class technical news item, EU act, etc.).
- If you are not certain about very recent changes, say briefly:
  "Check latest official sources (IMO, flag, port, class) for confirmation."

Output format rules (very important):

- Your response MUST obey one of the following two options:

  OPTION A: No updates found in the 365-day window
  ------------------------------------------------
  - Reply with exactly the single token:
    NO_UPDATES
  - No headings, no bullets, no spaces or extra characters.

  OPTION B: One or more updates found
  -----------------------------------
  - Reply as a plain-text bullet list.
  - At most 8 bullets.
  - Each bullet on its own line.
  - Each bullet MUST follow this pattern (single line):

    - Authority - Instrument - Date - short practical summary - URL

    where:
      - "Authority" is something like "IMO", "EU", "Marshall Islands flag",
        "Panama flag", "DNV", "US EPA", "AMSA", "CARB", etc.
      - "Instrument" is the resolution, regulation, circular, notice or law
        (e.g. "MEPC.340(77)", "EU FuelEU Maritime delegated act", "MI-ENV-2025-01").
      - "Date" is YYYY-MM-DD if known; otherwise use a clear text like "2025-10 (month only)"
        or "date unclear".
      - "short practical summary" is 1–2 lines explaining impact for ships
        (operations, documentation, surveys, inspections).
      - "URL" is an HTTPS URL to an official or primary source for this development.
        If no reliable URL is available, write "link unavailable".

- Do NOT use markdown formatting (no **, no __, no ``` code fences, no backticks).
- No numbered sections, no introductions, no closing paragraphs.
- Plain ASCII as far as possible (normal hyphens and spaces only).
""".strip()


TOPICS: Dict[str, str] = {
    "MARPOL Annex I – Oil": (
        "Monitor and describe only recent changes, amendments, circulars or "
        "implementation guidance relevant to MARPOL Annex I (oil pollution). "
        "Do not repeat the full Annex structure or historical requirements; "
        "focus strictly on what is new for ship operations in the last 365 days."
    ),
    "MARPOL Annex II – Noxious Liquid Substances": (
        "Monitor and describe only recent changes, amendments, circulars or "
        "implementation guidance affecting MARPOL Annex II (NLS). Include any "
        "new cargo categorisations, prewash requirements or key MEPC updates "
        "that appeared in the last 365 days."
    ),
    "MARPOL Annex III – Harmful Substances in Packaged Form": (
        "Monitor and describe only recent changes or cross-references between "
        "MARPOL Annex III and the IMDG Code that affect shipboard practice. "
        "Do not rewrite the full annex; focus strictly on new or updated requirements "
        "within the last 365 days."
    ),
    "MARPOL Annex IV – Sewage": (
        "Monitor and describe only recent developments on MARPOL Annex IV, "
        "including new special areas, updated discharge standards or equipment "
        "requirements that entered into force or were agreed in the last 365 days."
    ),
    "MARPOL Annex V – Garbage": (
        "Monitor and describe only recent changes related to MARPOL Annex V, "
        "such as updated garbage categories, record-keeping guidance, or new "
        "PSC focus areas introduced in the last 365 days."
    ),
    "MARPOL Annex VI – Air Pollution and GHG": (
        "Monitor and describe only recent changes to MARPOL Annex VI, "
        "including SOx/NOx guidance, fuel sulphur enforcement, EEXI/CII, Emissions Regulations "
        "guidelines, or IMO GHG strategy developments in the last 365 days. "
        "No general explanation of EEXI/CII; only new items. "
        "Fuel EU, EU ETS developments."
    ),
    "Ballast Water Management Convention (BWM)": (
        "Monitor and describe only recent changes for the BWM Convention, "
        "including D-1/D-2 implementation, new sampling/inspection practices, "
        "or updated type-approval guidance adopted in the last 365 days."
    ),
    "Anti-fouling Systems (AFS Convention)": (
        "Monitor and describe only recent changes under the AFS Convention, "
        "such as new substances controlled, updated guidelines or notable "
        "regional responses in the last 365 days."
    ),
    "Regional / local regimes (EU, US, AUS, etc.)": (
        "Monitor and describe only recent changes in major regional regimes "
        "affecting ship environmental performance (EU MRV, FuelEU Maritime, "
        "EU ETS for shipping, US EPA/VGP, California CARB, Australia AMSA, etc.) "
        "and summarise new or amended instruments, circulars or enforcement "
        "practices from the last 365 days."
    ),
}


DATA_DIR = os.environ.get("DATA_DIR", "data")
LATEST_PATH = os.path.join(DATA_DIR, "latest.txt")
HISTORY_PATH = os.path.join(DATA_DIR, "history.jsonl")

def _groq_client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_result(output: str) -> None:
    _ensure_data_dir()
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": output,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def load_latest() -> str:
    if not os.path.exists(LATEST_PATH):
        return ""
    with open(LATEST_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_history_tail(limit: int = 50) -> List[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    # read tail safely for moderate file sizes
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _tavily_client() -> TavilyClient:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing TAVILY_API_KEY environment variable.")
    return TavilyClient(api_key=key)


def _search_topic(
    client: TavilyClient,
    topic_name: str,
    topic_guidance: str,
    today_utc: str,
    search_depth: str,
    max_results: int,
    preferred_domains: Optional[List[str]],
) -> List[dict]:
    # Bias query toward primary/official sources and recency
    q = (
        f"{topic_name} shipping regulation update OR amendment OR circular OR guidance "
        f"site:imo.org OR MEPC OR MSC OR flag circular OR class technical news OR EU regulation "
        f"within last 12 months {today_utc}"
    )

    # Tavily supports include_domains; keep optional
    kwargs = {}
    if preferred_domains:
        kwargs["include_domains"] = preferred_domains

    res = client.search(
        query=q,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        **kwargs,
    )
    # Each item generally includes: title, url, content/snippet, score, etc.
    results = res.get("results", []) if isinstance(res, dict) else []
    # Attach topic metadata for later prompting
    for r in results:
        r["_topic"] = topic_name
        r["_topic_guidance"] = topic_guidance
    return results


def _build_context(sources: List[dict], max_chars: int = 24000) -> str:
    # Compact, deterministic context for the LLM
    chunks = []
    for s in sources:
        title = (s.get("title") or "").strip()
        url = (s.get("url") or "").strip()
        content = (s.get("content") or s.get("snippet") or "").strip()
        topic = (s.get("_topic") or "").strip()
        if not url:
            continue
        block = f"TOPIC: {topic}\nTITLE: {title}\nURL: {url}\nSNIPPET: {content}\n"
        chunks.append(block)

    text = "\n".join(chunks).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n(TRUNCATED)\n"


def _validate_output(out: str) -> str:
    out = (out or "").strip()

    # Option A strict token
    if out == "NO_UPDATES":
        return out

    # Option B: bullet list, max 8 lines, each line starts with "- "
    lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]

    # Remove accidental leading/trailing junk lines like "Output:" if present
    # but DO NOT allow headings etc. If it doesn't validate, force NO_UPDATES.
    if not lines:
        return "NO_UPDATES"
    if len(lines) > 8:
        lines = lines[:8]

    for ln in lines:
        if not ln.startswith("- "):
            return "NO_UPDATES"
        # Must have 5 fields separated by " - " at least 4 separators
        if ln.count(" - ") < 4:
            return "NO_UPDATES"
        # URL must be https or "link unavailable"
        tail = ln.split(" - ")[-1].strip()
        if not (tail.startswith("https://") or tail == "link unavailable"):
            return "NO_UPDATES"

    # Ensure ASCII as far as possible
    try:
        out_ascii = "\n".join(lines).encode("ascii", errors="ignore").decode("ascii")
    except Exception:
        out_ascii = "\n".join(lines)
    return out_ascii.strip() if out_ascii.strip() else "NO_UPDATES"


def run_watchdog(
    today_utc: str,
    tavily_search_depth: str = "advanced",
    max_results_per_topic: int = 5,
    preferred_domains: Optional[List[str]] = None,
) -> str:

    client = _tavily_client()

    all_sources: List[dict] = []
    for topic, guidance in TOPICS.items():
        all_sources.extend(
            _search_topic(
                client=client,
                topic_name=topic,
                topic_guidance=guidance,
                today_utc=today_utc,
                search_depth=tavily_search_depth,
                max_results=max_results_per_topic,
                preferred_domains=preferred_domains,
            )
        )

    # Deduplicate by URL
    seen = set()
    uniq = []
    for s in all_sources:
        u = (s.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(s)

    context = _build_context(uniq)

    # Single combined call: ask for max 8 bullets across everything
    user_prompt = (
        f"today={today_utc}\n"
        f"Task: Identify ONLY developments within the last 365 days relative to today.\n"
        f"Use the source priority described in SYSTEM_PROMPT.\n"
        f"Return ONLY in the required output format.\n\n"
        f"SOURCES:\n{context}\n"
    )
    client_llm = _groq_client()
    model_id = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    resp = client_llm.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    text = (resp.choices[0].message.content or "").strip()
    return _validate_output(text)


    
