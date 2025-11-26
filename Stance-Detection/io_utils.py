import os
import pandas as pd
from typing import Dict, Any, List

def ensure_header(out_csv: str, cols: List[str]) -> None:
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)

def load_done_ids(out_csv: str) -> set:
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return set()
    try:
        done = pd.read_csv(out_csv, usecols=["tweet_id"], dtype=str)
        return set(done["tweet_id"].dropna().astype(str).tolist())
    except Exception:
        return set()

def append_batch(out_csv: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out_csv, mode="a", header=False, index=False)
