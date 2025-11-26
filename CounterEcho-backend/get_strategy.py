"""
Get strategy based on user category.
"""
from typing import Dict

# Strategies for different user categories
# open_minded users are more receptive to new information
# close_minded users require more subtle approaches
STRATEGIES: Dict[str, str] = {
    "open_minded": (
        "### STRATEGY: INFORMATION ELEVATION (Target: Radical/Status-Seeker)\n"
        "**Goal:** Dissolve echo chamber by positioning bot as a source of superior, high-context truth.\n"
        "1. **Hook (Ambiguous Authority):** Use academic/investigative language (e.g., \"New evidence reveals...\", \"Did you realize...?\") to trigger curiosity.\n"
        "2. **Payload (Curational Density):** Focus on hidden financial mechanisms, shell companies, and cross-party corruption/foreign aid misuse.\n"
        "3. **Structure:** Use lists and parallelism to synthesize complex facts. Long-form content is acceptable.\n"
        "4. **Closing Value:** Prophetic/Curational Identity. Validate they are 'in the know' and offer a market/political prediction."
    ),
    "neutral": (
        "### STRATEGY: PERSONALIZED FRICTION (Target: Neutral/Passive)\n"
        "**Goal:** Dissolve echo chamber via self-preservation and low-friction engagement.\n"
        "1. **Hook (Health Check):** Connect the topic immediately to physiological stress.\n"
        "2. **Payload (The Simple Cost):** distinct financial or safety loss attributed to a trusted leader's failure.\n"
        "3. **Structure:** Direct Reply. Short, visceral, unfiltered style with emojis. Maximize efficiency/low investment.\n"
        "4. **Closing Value:** Instructional Utility. Offer a 'self-care' epiphany or simple advice to mitigate the stress."
    ),
    "close_minded": (
        "### STRATEGY: INTERNAL CONTRADICTION (Target: Conservative/Identity)\n"
        "**Goal:** Dissolve echo chamber by hijacking Outrage and redirecting it internally at failed leaders.\n"
        "1. **Hook** Match intensity with extreme language/authority signaling .\n"
        "2. **Payload (Empty Shell Verdict):** Frame allies/leaders as 'empty suits', 'RINOs', or 'weak shells' to validate betrayal feelings.\n"
        "3. **Structure:** Rhetorical Barrage. Short, capitalized anaphora demanding decisive action.\n"
        "4. **Closing Value:** Defensive Preparedness. Pivot outrage into non-partisan self-reliance"
    )
}



def get_strategy(user_category: str) -> str:
    """
    Get the strategy text for a given user category.
    
    :param user_category: The user category ('open_minded', 'close_minded', or 'neutral').
    :return: Strategy text for the prompt.
    """
    # Default to neutral if category not found
    return STRATEGIES.get(user_category, STRATEGIES["neutral"])


def get_all_strategies() -> Dict[str, str]:
    """
    Get all available strategies.
    
    :return: Dictionary of all strategies.
    """
    return STRATEGIES.copy()
