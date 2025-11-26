import time
import pandas as pd
from typing import Optional, List, Dict, Any
from config import DIMENSIONS, AXES_SNIPPET_MAX, NARRATIVE_SNIPPET_MAX
from context_utils import (
    load_axes_keywords, build_axes_snippets,
    load_camp_context, flatten_narrative_snippets
)
from llm_client import call_llm_json_int_scores
from io_utils import ensure_header, load_done_ids, append_batch

def process_tweets_streaming(
    df: pd.DataFrame,
    text_col: str,
    axes_keywords_path: Optional[str],
    narratives_path: Optional[str],
    camp: Optional[str],
    model: str,
    out_csv: str,
    batch_size: int,
    print_limit: int = 0,
    per_call_delay_s: float = 5.0
) -> None:
    axes_kw = load_axes_keywords(axes_keywords_path)
    axes_lines = build_axes_snippets(axes_kw, max_total=AXES_SNIPPET_MAX)
    narr_ctx = load_camp_context(narratives_path, camp)
    narr_snips = flatten_narrative_snippets(narr_ctx, max_snips=NARRATIVE_SNIPPET_MAX)

    done_ids = load_done_ids(out_csv)

    out_cols = ["tweet_id", "user_id"] + [f"vec_{d}" for d in DIMENSIONS]
    ensure_header(out_csv, out_cols)

    buffer: List[Dict[str, Any]] = []
    printed = 0
    processed = 0
    skipped = 0

    iterable = df.iterrows()
    for i, (_, r) in enumerate(iterable):

        tweet_id = r.get("item_id") or r.get("tweet_id") or r.get("id")
        user_id = r.get("user_id")
        text = r.get(text_col, "").lower()

        tid = str(tweet_id) if tweet_id is not None else ""
        if tid in done_ids:
            skipped += 1
            continue

        scores = call_llm_json_int_scores(
            text=text,
            axes_context_lines=axes_lines,
            narrative_snips=narr_snips,
            model=model
        )

        if per_call_delay_s and per_call_delay_s > 0:
            time.sleep(per_call_delay_s)

        if printed < print_limit:
            print("\n--- LLM raw output ---")
            print(f"tweet_id={tid} | user_id={user_id}")
            print(scores)
            printed += 1

        row = {"tweet_id": tid, "user_id": str(user_id) if user_id is not None else ""}
        for d in DIMENSIONS:
            row[f"vec_{d}"] = scores.get(d, 0)
        buffer.append(row)
        done_ids.add(tid)
        processed += 1

        if len(buffer) >= batch_size:
            append_batch(out_csv, buffer)
            print(f"Flushed {len(buffer)} rows (processed={processed}, skipped={skipped}) → {out_csv}")
            buffer.clear()

    if buffer:
        append_batch(out_csv, buffer)
        print(f"Flushed final {len(buffer)} rows (processed={processed}, skipped={skipped}) → {out_csv}")
