# Function to get user category from user_vectors_aggregated.csv
from pathlib import Path
import pandas as pd

def get_user_category(user_id: str):
	"""
	Returns (user_id, category) for the given user_id from user_vectors_aggregated.csv.
	Category: 'open_minded' if polarity between -0.8 and 0.8 (inclusive), else 'close_minded'.
	"""
	csv_path = "/scratch/ihmehta/SemanticWebMining/CounterEcho-AI/data/user_vectors_aggregated_categorized.csv"
	df = pd.read_csv(csv_path)
	row = df[df['user_id'] == user_id]
	if row.empty:
		return user_id, None
	
	polarity = row.iloc[0]['polarity']
	category = 'open_minded' if -0.9999999989950352 <= polarity  else 'close_minded'
	return user_id, category
