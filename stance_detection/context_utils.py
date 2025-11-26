import os, json
from typing import Dict, List, Any, Optional
from config import DIMENSIONS, AXES_BUCKET_LIMIT, AXES_SNIPPET_MAX, NARRATIVE_SNIPPET_MAX

def load_camp_context(narratives_path: Optional[str], camp: Optional[str]) -> Optional[Dict[str, Any]]:
    if not narratives_path or not os.path.exists(narratives_path):
        return None
    try:
        with open(narratives_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not camp:
        return data
    camp_keys = [k for k in data.keys() if k.upper().startswith(camp.upper()) or camp.upper() in k.upper()]
    return {k: data[k] for k in camp_keys} if camp_keys else data

def flatten_narrative_snippets(narr_ctx: Optional[Dict[str, Any]], max_snips: int = NARRATIVE_SNIPPET_MAX) -> List[str]:
    if not narr_ctx:
        return []
    snippets = []
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and ("{" in v or "}" in v or "," in v or len(v) <= 160):
                    snippets.append(f"{k}: {v}")
                else:
                    walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)
    walk(narr_ctx)
    seen, out = set(), []
    for s in snippets:
        s2 = s.strip().lower()
        if s2 and s2 not in seen:
            out.append(s2[:220])
            seen.add(s2)
        if len(out) >= max_snips:
            break
    return out

def load_axes_keywords(path: Optional[str]) -> Optional[Dict[str, Dict[str, List[str]]]]:
    if not path or not os.path.exists(path):
        print("Warning: axes_keywords.json path not provided or does not exist.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = {}
    for axis in DIMENSIONS:
        axis_dict = data.get(axis, {}) or {}
        out_axis = {}
        for bucket in ("support", "oppose", "neutral_cues"):
            items = axis_dict.get(bucket, []) or []
            norm, seen = [], set()
            for it in items:
                s = (it or "").strip().lower()
                if not s or s in seen:
                    continue
                seen.add(s)
                norm.append(s)
                if len(norm) >= AXES_BUCKET_LIMIT:
                    break
            out_axis[bucket] = norm
        cleaned[axis] = out_axis
    return cleaned

def build_axes_snippets(axes_kw: Optional[Dict[str, Dict[str, List[str]]]], max_total: int = AXES_SNIPPET_MAX) -> List[str]:
    if not axes_kw:
        return []
    lines = []
    for axis in DIMENSIONS:
        buckets = axes_kw.get(axis, {})
        for bucket in ("support", "oppose", "neutral_cues"):
            phrases = buckets.get(bucket, [])
            if not phrases:
                continue
            frag = "; ".join(phrases[:10])
            lines.append(f"{axis}.{bucket}: {frag}")
            if len(lines) >= max_total:
                return lines
    return lines
