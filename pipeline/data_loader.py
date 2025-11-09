"""
Data Loading Module

This module handles loading of graph, community, and tweet data.

Community-Camp Mapping:
- RU/CH CAMP: [C3, C7, C21, C12]
- LEFT CAMP: [C25, C10, C1, C5]
- RIGHT CAMP: [C2, C6]
"""

import pandas as pd
import networkx as nx
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def load_graph_and_communities(graph_path: str, community_path: str) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Load the retweet network graph and community assignments.
    
    Args:
        graph_path: Path to all.gexf
        community_path: Path to community.csv
        
    Returns:
        Tuple of (graph, community_df)
    """
    logger.info("Loading data files...")
    
    logger.info(f"Loading graph from {graph_path}...")
    G = nx.read_gexf(graph_path)
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    logger.info(f"Loading communities from {community_path}...")
    community_df = pd.read_csv(community_path)
    logger.info(f"Communities loaded: {len(community_df)} users")
    
    return G, community_df


def load_tweets(tweets_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load tweet dataset with optional sampling.
    
    Args:
        tweets_path: Path to cleaned_dataset.csv
        sample_size: Number of tweets to sample (None for all)
        
    Returns:
        DataFrame with tweets
    """
    logger.info(f"Loading tweets from {tweets_path}...")
    
    if sample_size:
        logger.info(f"Sampling {sample_size} tweets for performance...")
        chunks = []
        for chunk in pd.read_csv(tweets_path, chunksize=100000):
            chunks.append(chunk)
            if sum(len(c) for c in chunks) >= sample_size:
                break
        tweets_df = pd.concat(chunks, ignore_index=True).sample(n=min(sample_size, sum(len(c) for c in chunks)), random_state=42)
    else:
        tweets_df = pd.read_csv(tweets_path)
    
    logger.info(f"Tweets loaded: {len(tweets_df)} tweets")
    
    return tweets_df
