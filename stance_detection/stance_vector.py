# stance_vector.py
import os
import argparse
import pandas as pd
from config import DEFAULT_MODEL, DIMENSIONS
from processor import process_tweets_streaming

def main():
    ap = argparse.ArgumentParser(
        description="Integer stance vectors for tweets using axes_keywords.json context (batched writes, resume-safe)"
    )
    ap.add_argument("--tweets", required=True, help="Path to tweets CSV (expects item_id,user_id,content,post_type columns).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (append-safe; resume supported).")
    ap.add_argument("--tweets_col", default="content", help="Text column name (default: content).")
    ap.add_argument("--axes_keywords", default="./axes_keywords.json", help="Path to axes_keywords.json.")
    ap.add_argument("--narratives", default=None, help="Optional narratives.json (legacy).")
    ap.add_argument("--camp", default=None, help="Optional camp key (RIGHT/LEFT/RU_CH).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LLM model (e.g., gemini-1.5-flash).")
    ap.add_argument("--print_limit", type=int, default=0, help="Print first N raw LLM JSON outputs.")
    ap.add_argument("--batch_size", type=int, default=10, help="How many rows to buffer before flushing to CSV.")
    ap.add_argument("--per_call_delay_s", type=float, default=5.0, help="Sleep after each LLM call (seconds).")
    args = ap.parse_args()

    if not os.path.exists(args.tweets):
        raise FileNotFoundError(f"Tweets file not found: {args.tweets}")

    df = pd.read_csv(args.tweets)
    needed = [args.tweets_col, "user_id"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {args.tweets}")
    if "item_id" not in df.columns and "tweet_id" not in df.columns and "id" not in df.columns:
        raise ValueError("No tweet id column found (expected one of: item_id, tweet_id, id).")

    process_tweets_streaming(
        df=df,
        text_col=args.tweets_col,
        axes_keywords_path=args.axes_keywords,
        narratives_path=args.narratives,
        camp=args.camp,
        model=args.model,
        out_csv=args.out_csv,
        batch_size=max(1, args.batch_size),
        print_limit=args.print_limit,
        per_call_delay_s=max(0.0, args.per_call_delay_s)
    )

    print(f"\nDone. Output appended to: {args.out_csv}")

if __name__ == "__main__":
    main()
