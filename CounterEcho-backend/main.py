"""
FastAPI webserver for CounterEcho-AI.
Takes user_id and tweet_content as input and generates a counter message.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from stance_detection
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from enum import Enum
import pandas as pd
import logging

# Local imports
from get_user_category import get_user_category
from get_strategy import get_strategy
from get_metadata import get_stances_to_counter, get_counter_narratives_for_stances, format_counter_narratives_for_prompt
from prompt_generator import build_counterecho_prompt
from llm_generation import generate_counter_message
from stance_detection import call_llm_json_int_scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CounterEcho-AI API",
    description="API for generating counter messages to break echo chambers",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_DATASET_PATH = DATA_DIR / "enriched_dataset_with_vectors.csv"
COMMUNITY_CSV_PATH = DATA_DIR / "community.csv"

# Community to Camp mapping
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

# Stance dimensions
DIMENSIONS = ["LNG", "BIG_OIL", "RENEWABLES", "RUSSIA", "CHINA", "US_POLICY", "EU_UKRAINE", "SANCTIONS_TARIFFS", "ISRAEL"]


class TweetType(str, Enum):
    """Tweet type enum."""
    NEW = "NEW"
    EXISTING = "EXISTING"


class CampType(str, Enum):
    """Camp type enum."""
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    RU_CH = "RU_CH"


class CounterMessageRequest(BaseModel):
    """Request model for counter message generation - matches frontend format."""
    tweet_type: TweetType = Field(..., alias="tweet_type")
    content: str
    camp: CampType
    userId: str


class CounterMessageResponse(BaseModel):
    """Response model for counter message generation - matches frontend format."""
    content: str
    camp: str
    userId: str
    counterMessage: str
    userType: Optional[str] = None  # "closed_minded" or "open_minded"
    error: Optional[str] = None


# Keep old models for backward compatibility
class LegacyCounterMessageRequest(BaseModel):
    """Legacy request model for counter message generation."""
    user_id: str
    tweet_content: str


class LegacyCounterMessageResponse(BaseModel):
    """Legacy response model for counter message generation."""
    counter_message: str
    stance_vector: Dict[str, int]
    counter_stance_vector: Dict[str, int]
    user_category: Optional[str]
    user_camp: Optional[str]
    debug_info: Optional[Dict[str, Any]] = None


class StanceDetectionRequest(BaseModel):
    """Request model for stance detection only."""
    tweet_content: str


class StanceDetectionResponse(BaseModel):
    """Response model for stance detection."""
    stance_vector: Dict[str, int]


def get_user_camp_from_enriched_dataset(user_id: str) -> Optional[str]:
    """
    Get user camp from enriched_dataset_with_vectors.csv via community.
    
    :param user_id: The user ID (e.g., 'tw123456').
    :return: Camp identifier or None if not found.
    """
    try:
        # First try to find the user in the enriched dataset to get their community
        enriched_df = pd.read_csv(ENRICHED_DATASET_PATH)
        
        # Look for user in user_id column
        user_rows = enriched_df[enriched_df['user_id'] == user_id]
        
        if not user_rows.empty:
            # Get the community from the first matching row
            community = user_rows.iloc[0].get('community')
            if community and community in COMMUNITY_TO_CAMP:
                return COMMUNITY_TO_CAMP[community]
        
        # Fallback: check community.csv directly
        community_df = pd.read_csv(COMMUNITY_CSV_PATH)
        user_row = community_df[community_df['Id'] == user_id]
        
        if not user_row.empty:
            community = user_row.iloc[0]['community']
            if community in COMMUNITY_TO_CAMP:
                return COMMUNITY_TO_CAMP[community]
        
        return None
    except Exception as e:
        logger.error(f"Error getting user camp: {e}")
        return None


def negate_stance_vector(stance_vector: Dict[str, int]) -> Dict[str, int]:
    """
    Negate the stance vector to get counter stances.
    
    :param stance_vector: Original stance vector.
    :return: Negated stance vector.
    """
    return {k: -v for k, v in stance_vector.items()}


def detect_stance(tweet_content: str) -> Dict[str, int]:
    """
    Detect stance from tweet content using LLM.
    
    :param tweet_content: The tweet text.
    :return: Dictionary of stance values.
    """
    # Call the stance detection LLM
    stance_vector = call_llm_json_int_scores(
        text=tweet_content,
        axes_context_lines=[],
        narrative_snips=[],
    )
    return stance_vector


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CounterEcho-AI API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/detect_stance", response_model=StanceDetectionResponse)
async def detect_stance_endpoint(request: StanceDetectionRequest):
    """
    Detect stance from a tweet.
    
    :param request: Request containing tweet_content.
    :return: Stance vector.
    """
    try:
        stance_vector = detect_stance(request.tweet_content)
        return StanceDetectionResponse(stance_vector=stance_vector)
    except Exception as e:
        logger.error(f"Error detecting stance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_counter_message", response_model=CounterMessageResponse)
async def generate_counter_message_endpoint(request: CounterMessageRequest):
    """
    Generate a counter message for a tweet.
    
    Flow:
    1. Detect stance vector from tweet
    2. Negate the stance vector
    3. Get user category from get_user_category
    4. Use camp from request (or lookup from dataset)
    5. Get narrative metadata for the counter stances
    6. Build prompt and generate counter message
    
    :param request: Request containing tweet_type, content, camp, and userId.
    :return: Counter message and metadata in frontend format.
    """
    try:
        logger.info(f"Processing request for userId: {request.userId}")
        
        # Step 1: Detect stance from tweet
        logger.info("Step 1: Detecting stance...")
        stance_vector = detect_stance(request.content)
        logger.info(f"Detected stance: {stance_vector}")
        
        # Step 2: Identify stances to counter based on trigger conditions
        logger.info("Step 2: Identifying stances to counter...")
        stances_to_counter = get_stances_to_counter(stance_vector)
        logger.info(f"Stances to counter: {stances_to_counter}")
        
        # Step 3: Negate the stance vector (for reference)
        counter_stance_vector = negate_stance_vector(stance_vector)
        
        # Step 4: Get user category
        logger.info("Step 3: Getting user category...")
        _, user_category = get_user_category(request.userId)
        
        logger.info(f"User category: {user_category}")
        
        # Step 5: Use camp from request
        logger.info("Step 4: Using camp from request...")
        user_camp = request.camp.value
        logger.info(f"User camp: {user_camp}")
        
        # Step 6: Get strategy for user category
        logger.info("Step 5: Getting strategy...")
        strategy = get_strategy(user_category)
        logger.info(f"Strategy retrieved for category: {user_category}")
        
        # Step 7: Get relevant counter-narratives (only for stances we're countering)
        logger.info("Step 6: Getting counter-narratives...")
        counter_narratives = get_counter_narratives_for_stances(stances_to_counter, user_camp)
        counter_narratives_str = format_counter_narratives_for_prompt(counter_narratives)
        logger.info(f"Counter-narratives retrieved for {len(counter_narratives)} dimensions")
        
        # Step 8: Build prompt
        logger.info("Step 7: Building prompt...")
        prompt = build_counterecho_prompt(
            tweet=request.content,
            stance_vector=stance_vector,
            counter_stance_vector=counter_stance_vector,
            user_category=user_category,
            user_camp=user_camp,
            strategy=strategy,
            counter_narratives_str=counter_narratives_str,
            stances_to_counter=stances_to_counter
        )
        logger.info(f"The complete prompt length is {len(prompt)} characters.")
        logger.info(f"Prompt preview: {prompt}...")
        # Step 9: Generate counter message
        logger.info("Step 8: Generating counter message...")
        try:
            counter_message = generate_counter_message(prompt)
        except Exception as e:
            logger.warning(f"Compound model failed, falling back to simple model: {e}")
            # counter_message = generate_counter_message_simple(prompt)
        
        logger.info(f"Generated counter message: {counter_message[:1000] if counter_message else 'empty'}...")
        
        # Map user_category to userType format expected by frontend
        user_type = "closed_minded" if user_category == "closed_minded" else "open_minded" if user_category == "open_minded" else "open_minded"
        
        return CounterMessageResponse(
            content=request.content,
            camp=user_camp,
            userId=request.userId,
            counterMessage=counter_message,
            userType=user_type,
            error=None
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CounterMessageResponse(
            content=request.content,
            camp=request.camp.value,
            userId=request.userId,
            counterMessage="",
            userType=None,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating counter message: {e}")
        return CounterMessageResponse(
            content=request.content,
            camp=request.camp.value,
            userId=request.userId,
            counterMessage="",
            userType=None,
            error=str(e)
        )


@app.get("/user/{user_id}/info")
async def get_user_info(user_id: str):
    """
    Get information about a user.
    
    :param user_id: The user ID.
    :return: User category and camp.
    """
    try:
        _, user_category = get_user_category(user_id)
        user_camp = get_user_camp_from_enriched_dataset(user_id)
        
        return {
            "user_id": user_id,
            "user_category": user_category,
            "user_camp": user_camp
        }
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
