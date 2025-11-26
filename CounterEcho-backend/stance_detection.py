import os, json, time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Any
from stance_detection.config import (
    DIMENSIONS, DEFAULT_MODEL, PROMPT_AXES_LINES_LIMIT, PROMPT_NARRATIVE_LINES_LIMIT
)
import google.generativeai as genai

def _extract_json_object(txt: str) -> Dict[str, Any]:
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(txt[start:end+1])
    return json.loads(txt)

def _coerce_int_scores(js: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for d in DIMENSIONS:
        v = js.get(d, 0)
        try:
            v = int(round(float(v)))
        except Exception:
            v = 0
        if v < -1: v = -1
        if v >  1: v =  1
        out[d] = v
    return out

def call_llm_json_int_scores(
    text: str,
    axes_context_lines: List[str],
    narrative_snips: List[str],
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    sleep_s: float = 2,
    post_delay_s: float = 0  # set >0 if you want a fixed pause after every call
) -> Dict[str, int]:
    """
    Return integer scores {-1,0,1} per axis.
    Tries Gemini (env GEMINI_API_KEY), then Groq (env GROQ_API_KEY), else zeros.
    """
    sys_instructions = (
        f"Assign INTEGER stance scores for the tweet across {len(DIMENSIONS)} axes.\n"
        "Return ONLY a compact JSON object with keys exactly: "
        + ", ".join(DIMENSIONS) +
        ". Each value MUST be one of: -1, 0, or 1 (integers only). "
        "-1=oppose, 0=neutral/irrelevant, 1=support. "
        "If a dimension is not mentioned, use 0. No explanations."
    )

    ctx_axes = ""
    if axes_context_lines:
        ctx_axes = "Axis keyword cues (for disambiguation; not exhaustive):\n- " + \
                   "\n- ".join(axes_context_lines[:PROMPT_AXES_LINES_LIMIT])

    ctx_narr = ""
    if narrative_snips:
        ctx_narr = "Legacy narrative snippets:\n- " + \
                   "\n- ".join(narrative_snips[:PROMPT_NARRATIVE_LINES_LIMIT])

    user_prompt = f"""Tweet:
\"\"\"{(text or '').strip()[:4000]}\"\"\"

Axes (decide stance per axis independently):
- LNG (US LNG / natural gas exports, pipelines, fracking)
- BIG_OIL (oil & gas companies)
- RENEWABLES (solar, wind, net-zero)
- RUSSIA (reliance/condemnation re: energy)
- CHINA (energy trade/tariffs/sabotage/demand)
- US_POLICY (US domestic energy policy, prices, permits)
- EU_UKRAINE (EU energy security, Ukraine transit/support)
- SANCTIONS_TARIFFS (support/opposition to sanctions/tariffs in energy context)
- ISRAEL (support/oppose the state of Israel or its energy/geopolitical interests)

{ctx_axes}

{ctx_narr}

Respond with JSON like:
{{"LNG": 1, "BIG_OIL": -1, "RENEWABLES": -1, "RUSSIA": 0, "CHINA": 0, "US_POLICY": 1, "EU_UKRAINE": 0, "SANCTIONS_TARIFFS": 1, "ISRAEL": 0}}"""

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gen_model = genai.GenerativeModel(model)
        for attempt in range(max_retries):
            try:
                if attempt:
                    time.sleep(sleep_s * attempt)
                resp = gen_model.generate_content(
                    [{"role": "user", "parts": [{"text": sys_instructions + "\n\n" + user_prompt}]}],
                    safety_settings=None,
                )
                txt = (resp.candidates[0].content.parts[0].text or "").strip()
                js = _extract_json_object(txt)
                return _coerce_int_scores(js)
            except Exception as e:
                print(f"Gemini LLM call failed on attempt {attempt+1}/{max_retries}. Error {e}.")
                if attempt == max_retries - 1:
                    break
        return {d: 0 for d in DIMENSIONS}
    finally:
        if post_delay_s and post_delay_s > 0:
            time.sleep(post_delay_s)
