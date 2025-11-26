# Function to get user category from user_vectors_aggregated.csv
from pathlib import Path
import pandas as pd

def get_user_category(user_id: str):
	"""
	Returns (user_id, category) for the given user_id from user_vectors_aggregated.csv.
	Category: 'neutral' if polarity between -0.6 and 0.6 (inclusive), else 'radical'.
	"""
	csv_path = Path(__file__).resolve().parent / "user_ranking_data" / "user_vectors_aggregated.csv"
	df = pd.read_csv(csv_path)
	row = df[df['user_id'] == user_id]
	if row.empty:
		return user_id, None
	polarity = row.iloc[0]['polarity']
	if -0.6 <= polarity <= 0.6:
		category = 'neutral'
	else:
		category = 'radical'
	return user_id, category

