"""
Fetch metadata about stances from narratives.json.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# Path to narratives.json
NARRATIVES_PATH = Path(__file__).parent.parent / "data" / "narratives.json"

# Define which stance values we want to counter
# Key: dimension, Value: the stance value that triggers countering
# e.g., if BIG_OIL is -1 in the tweet, we want to counter it (promote pro-BIG_OIL)
COUNTER_TRIGGER_CONDITIONS = {
    "BIG_OIL": -1,       # Counter when tweet is anti-BIG_OIL
    "CHINA": 1,          # Counter when tweet is pro-CHINA
    "EU_UKRAINE": -1,    # Counter when tweet is anti-EU_UKRAINE
    "ISRAEL": -1,        # Counter when tweet is anti-ISRAEL
    "LNG": -1,           # Counter when tweet is anti-LNG
    "RENEWABLES": -1,    # Counter when tweet is anti-RENEWABLES (we counter by being pro-LNG instead)
    "RUSSIA": 1,         # Counter when tweet is pro-RUSSIA
    "SANCTIONS_TARIFFS": -1,  # Counter when tweet is anti-SANCTIONS_TARIFFS
    "US_POLICY": -1,     # Counter when tweet is anti-US_POLICY
}

# Mapping from stance dimension to narrative topic keys in narratives.json
# These are the topics we use to BUILD our counter-narrative
DIMENSION_TO_COUNTER_TOPICS = {
    "LNG": {
        "RU_CH CAMP": ["anti-US_USA_America"],  # They use anti-US narratives, we counter with pro-LNG
        "LEFT CAMP": ["anti-US_USA_America", "pro-US_USA_America"],
        "RIGHT CAMP": ["pro-LNG_Gas_Fracking:"],
    },
    "BIG_OIL": {
        "RU_CH CAMP": ["anti-IOC_Big_Oil", "pro-IOC"],
        "LEFT CAMP": ["anti-IOC_Big_Oil", "pro-IOC Themes (Less Prominent and Often Countered):"],
        "RIGHT CAMP": ["pro-LNG_Gas_Fracking:"],
    },
    "RENEWABLES": {
        "RU_CH CAMP": [],
        "LEFT CAMP": ["anti-IOC_Big_Oil"],
        "RIGHT CAMP": ["anti-Renewables_Green-Energy_Net-Zero:", "pro-LNG_Gas_Fracking:"],
    },
    "RUSSIA": {
        "RU_CH CAMP": ["pro-RUSSIA", "anti-RUSSIA"],
        "LEFT CAMP": ["anti-RUSSIA:", "pro-Russia"],
        "RIGHT CAMP": ["anti-Russia:"],
    },
    "CHINA": {
        "RU_CH CAMP": ["pro-CHINA", "anti-CHINA"],
        "LEFT CAMP": ["pro-China", "anti-China"],
        "RIGHT CAMP": ["anti-China:"],
    },
    "US_POLICY": {
        "RU_CH CAMP": ["anti-US_USA_America"],
        "LEFT CAMP": ["anti-US_USA_America", "pro-US_USA_America"],
        "RIGHT CAMP": ["pro-LNG_Gas_Fracking:"],
    },
    "EU_UKRAINE": {
        "RU_CH CAMP": ["anti-RUSSIA"],
        "LEFT CAMP": ["pro-EU_Ukraine", "anti-Europe_Ukraine", "pro-Ukraine"],
        "RIGHT CAMP": ["pro-EU_Ukraine:", "anti-EU:", "anti-Ukraine:"],
    },
    "SANCTIONS_TARIFFS": {
        "RU_CH CAMP": ["anti-US_USA_America"],
        "LEFT CAMP": ["anti-US_USA_America"],
        "RIGHT CAMP": ["pro-LNG_Gas_Fracking:"],
    },
    "ISRAEL": {
        "RU_CH CAMP": ["anti-ISRAEL"],
        "LEFT CAMP": [],
        "RIGHT CAMP": [],
    },
}

# Camp names in narratives.json
CAMP_NAMES = {
    "RU_CH": "RU_CH CAMP",
    "LEFT": "LEFT CAMP",
    "RIGHT": "RIGHT CAMP"
}


def load_narratives() -> Dict[str, Any]:
    """Load narratives.json and return as dictionary."""
    with open(NARRATIVES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_stances_to_counter(stance_vector: Dict[str, int]) -> Dict[str, int]:
    """
    Identify which stances from the tweet we want to counter.
    
    Only returns dimensions where the stance matches our counter trigger conditions.
    
    :param stance_vector: Dictionary of stance values detected in the tweet.
    :return: Dictionary of dimensions to counter with their original stance values.
    """
    to_counter = {}
    for dimension, trigger_value in COUNTER_TRIGGER_CONDITIONS.items():
        detected_value = stance_vector.get(dimension, 0)
        if detected_value == trigger_value:
            to_counter[dimension] = detected_value
    return to_counter


def get_counter_narratives_for_stances(stances_to_counter: Dict[str, int], camp: str) -> Dict[str, Any]:
    """
    Get relevant narratives from narratives.json for the stances we want to counter.
    
    :param stances_to_counter: Dictionary of dimensions to counter (from get_stances_to_counter).
    :param camp: The camp identifier ('RU_CH', 'LEFT', or 'RIGHT').
    :return: Dictionary with narrative metadata for each dimension to counter.
    """
    narratives = load_narratives()
    camp_key = CAMP_NAMES.get(camp, camp)
    
    # Get camp-specific narratives
    camp_data = narratives.get(camp_key, {})
    topics = camp_data.get("topics", {})
    
    result = {}
    
    for dimension, stance_value in stances_to_counter.items():
        # Get the topic keys relevant for countering this dimension in this camp
        topic_keys = DIMENSION_TO_COUNTER_TOPICS.get(dimension, {}).get(camp_key, [])
        
        dimension_narratives = {}
        for topic_key in topic_keys:
            if topic_key in topics:
                topic_data = topics[topic_key]
                if "narratives" in topic_data:
                    dimension_narratives[topic_key] = topic_data["narratives"]
        
        if dimension_narratives:
            # Determine counter stance type (opposite of detected)
            counter_type = "pro" if stance_value < 0 else "anti"
            result[dimension] = {
                "original_stance": stance_value,
                "counter_type": counter_type,
                "narratives": dimension_narratives
            }
    
    return result


def format_counter_narratives_for_prompt(counter_narratives: Dict[str, Any]) -> str:
    """
    Format counter-narrative metadata into a string suitable for the prompt.
    Only includes narrative names/titles, not the full keyword patterns.
    
    :param counter_narratives: Dictionary from get_counter_narratives_for_stances.
    :return: Formatted string with narrative names only.
    """
    if not counter_narratives:
        return "No specific narratives available for countering."
    
    lines = []
    for dimension, data in counter_narratives.items():
        counter_type = data.get("counter_type", "unknown")
        original = "pro" if data.get("original_stance", 0) > 0 else "anti"
        lines.append(f"\n### Counter {original.upper()}-{dimension} (use {counter_type.upper()}-{dimension} narratives):")
        
        for topic_key, narr_dict in data.get("narratives", {}).items():
            lines.append(f"\n**{topic_key}:**")
            # Only include narrative names, not the full keyword patterns
            narrative_names = list(narr_dict.keys())
            for narr_name in narrative_names:
                lines.append(f"  - {narr_name}")
    
    return "\n".join(lines)


def get_all_camp_narratives(camp: str) -> Dict[str, Any]:
    """
    Get all narratives for a specific camp.
    
    :param camp: The camp identifier ('RU_CH', 'LEFT', or 'RIGHT').
    :return: Dictionary with all narratives for the camp.
    """
    narratives = load_narratives()
    camp_key = CAMP_NAMES.get(camp, camp)
    return narratives.get(camp_key, {})
