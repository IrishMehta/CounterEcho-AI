from pathlib import Path
import pandas as pd
import numpy as np

# === 1. Load the dataset ===
# Use relative path to actual data location
input_csv = Path(__file__).resolve().parent / "tweet_vector_data" / "enriched_dataset_with_vectors.csv"
df = pd.read_csv(input_csv)

# === 2. Identify topic vector columns ===
vec_cols = [c for c in df.columns if c.startswith("vec_")]
vec_cols.sort()
print("Vector columns:", vec_cols)

# === 3. Aggregate per-user mean vector ===
user_vectors = df.groupby("user_id", sort=False)[vec_cols].mean().fillna(0.0)

# === 4. L2-normalize rows ===
def l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

user_vectors_l2 = pd.DataFrame(
    l2_normalize_rows(user_vectors.values),
    index=user_vectors.index,
    columns=user_vectors.columns
)

# === 5. Prepare bipartite matrix for ANCO-HITS ===
A = user_vectors.values  # users × topics

# === 6. Modified ANCO-HITS ===
def anco_hits(A, tol=1e-6, max_iter=1000, eps=1e-9):
    m, n = A.shape
    X = np.ones(m)
    Y = np.ones(n)
    for _ in range(max_iter):
        X_new = np.array([
            np.sum(A[i, :] * Y) / (np.sum(np.abs(A[i, :] * Y)) + eps)
            for i in range(m)
        ])
        Y_new = np.array([
            np.sum(A[:, j] * X_new) / (np.sum(np.abs(A[:, j] * X_new)) + eps)
            for j in range(n)
        ])
        if np.linalg.norm(X_new - X) + np.linalg.norm(Y_new - Y) < tol:
            break
        X, Y = X_new, Y_new
    return np.clip(X, -1, 1), np.clip(Y, -1, 1)

X, Y = anco_hits(A)

# === 7. Results ===
df_users_out = pd.DataFrame({
    "user_id": user_vectors.index,
    "polarity": X,
}).set_index("user_id").join(user_vectors)

df_topics_out = pd.DataFrame({
    "topic": user_vectors.columns,
    "polarity": Y
}).set_index("topic")

# === 8. Save outputs ===
output_dir = Path(__file__).resolve().parent / "user_ranking_data"
output_dir.mkdir(parents=True, exist_ok=True)
df_users_out.reset_index().to_csv(output_dir / "user_vectors_aggregated.csv", index=False)
user_vectors_l2.reset_index().to_csv(output_dir / "user_vectors_aggregated_l2.csv", index=False)
df_topics_out.reset_index().to_csv(output_dir / "anco_hits_results.csv", index=False)

print(f"Aggregated user vectors and computed ANCO-HITS successfully. Outputs written to: {output_dir}")
