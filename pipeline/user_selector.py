"""
User Selection Module

This module handles random user selection from the network.

Community-Camp Mapping:
- RU/CH CAMP: [C3, C7, C21, C12]
- LEFT CAMP: [C25, C10, C1, C5]
- RIGHT CAMP: [C2, C6]
"""

import pandas as pd
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

COMMUNITY_TO_CAMP = {
    'C3': 'RU_CH',
    'C7': 'RU_CH',
    'C21': 'RU_CH',
    'C12': 'RU_CH',
    'C25': 'LEFT',
    'C10': 'LEFT',
    'C1': 'LEFT',
    'C5': 'LEFT',
    'C2': 'RIGHT',
    'C6': 'RIGHT',
}


def select_random_user_with_tweet(
    community_df: pd.DataFrame, 
    tweets_df: pd.DataFrame
) -> Tuple[str, str, str, str]:
    """
    Select a random user from the network with their actual tweet.
    
    Args:
        community_df: DataFrame with Id and community columns
        tweets_df: DataFrame with user_id and content columns
        
    Returns:
        Tuple of (user_id, community, camp, tweet_text)
    """
    logger.debug("Converting user IDs to string format")
    tweets_df['user_id'] = tweets_df['user_id'].astype(str)
    community_df['Id'] = community_df['Id'].astype(str)
    
    users_with_tweets = set(tweets_df['user_id'].unique())
    community_with_tweets = community_df[community_df['Id'].isin(users_with_tweets)]
    
    logger.info(f"Found {len(community_with_tweets)} users with both community and tweets")
    
    if len(community_with_tweets) == 0:
        logger.error("No users found with both community assignment and tweets")
        raise ValueError("No users found with both community assignment and tweets")
    
    selected_row = community_with_tweets.sample(n=1).iloc[0]
    user_id = str(selected_row['Id'])
    community = selected_row['community']
    
    logger.debug(f"Selected user {user_id} from community {community}")
    
    camp = COMMUNITY_TO_CAMP.get(community)
    if not camp:
        logger.error(f"Community {community} not mapped to any camp")
        raise ValueError(f"Community {community} not mapped to any camp")
    
    user_tweets = tweets_df[tweets_df['user_id'] == user_id]
    if len(user_tweets) == 0:
        logger.error(f"No tweets found for user {user_id}")
        raise ValueError(f"No tweets found for user {user_id}")
    
    logger.debug(f"User has {len(user_tweets)} tweets available")
    selected_tweet = user_tweets.sample(n=1).iloc[0]
    tweet_text = str(selected_tweet['content'])
    
    return user_id, community, camp, tweet_text
