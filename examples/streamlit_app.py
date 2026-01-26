from __future__ import annotations

import streamlit as st

from moose_ui.config import sidebar
from moose_ui.metadata import fetch_assets, fetch_privacy_profiles

st.set_page_config(page_title="Moose", layout="wide")
cfg = sidebar()

st.title("Moose UI")

# Optional: show a quick connection/status summary
st.caption(f"Connected to: {cfg.get('base_url')}")

st.markdown(
    """
Moose helps you run annotation endpoints on **text** and **tables**.  
Jobs are asynchronous: you **submit** work, then **auto-poll** and render results in-place (recommended), while also saving job IDs in history.

---

## Column typing vs Cell NER vs CPA (tables)

### Column typing (`/schemas/{schema}/tabular/annotate`)
- **One label per column** using header + value distribution.
- Best for structured columns (email, ip, phone, ids, dates).
- Examples:
  - `email` → `ext:email` (STI)
  - `email` → `dpv-pd:EmailAddress` (DPV personal data)

### Cell NER (`/schemas/{schema}/tabular/ner`)
- Extracts entities **inside cells** for selected columns (`target_columns`).
- Best for free-text columns (notes, comments, descriptions).
- Example: `"Contact alice@example.com"` → entity `alice@example.com`

### CPA (`/schemas/{schema}/tabular/cpa`)
- Predicts **relationships** from a **subject column** to each target column.
- Fast CPA: `schema=cpa` (small generic label set)
- Deep CPA: `schema=schemaorg_cpa_v1` (schema.org predicates + `moose:NONE`, supports domain filtering and optional debug)
- Example: `BookName → Language` → `schema:inLanguage`

---

## Privacy Analyze (`/privacy/analyze`)
Privacy is a pipeline endpoint that returns:
- findings (issues + severity + recommended actions)
- optional legal references (GDPRtEXT concept refs)
- optional extraction output (NER/typing details)

Moose supports **profiles** (fast/balanced/deep) to control speed vs accuracy without changing endpoints.

---

## Which one should I use?
- If you have a **table** and want **types per column** → *Tables → Column typing*
- If you have a **table** and want **entities inside text cells** → *Tables → Cell NER*
- If you have a **table** and want **predicates/relationships** → *Tables → CPA*
- If you want **privacy findings + recommendations (+ optional GDPR refs)** → *Privacy*

---

## Pages
Use the pages on the left:

- **Text**: run NER over text
- **Tables**: typing, cell NER, CPA (with optional debug)
- **Privacy**: profile-driven privacy analysis (with optional legal refs)
- **Jobs**: poll and view results
- **Developer**: metadata + diagnostics (enable Developer mode in sidebar)
"""
)

# Optional: show available profiles + assets summary (non-blocking)
if cfg.get("api_key"):
    with st.expander("System summary (profiles/assets)", expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Privacy profiles")
            try:
                profiles_payload = fetch_privacy_profiles(cfg["base_url"], cfg["api_key"])
                st.json(
                    {
                        "default_profile": profiles_payload.get("default_profile"),
                        "profiles": list((profiles_payload.get("profiles") or {}).keys()),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load /privacy/profiles: {exc}")

        with col2:
            st.markdown("### Assets index")
            try:
                assets = fetch_assets(cfg["base_url"], cfg["api_key"])
                # show just top-level keys to avoid dumping the whole thing
                st.json({"keys": sorted(list(assets.keys()))})
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load /assets: {exc}")
else:
    st.info("Enter your API key in the sidebar to enable metadata (profiles/assets) summary.")