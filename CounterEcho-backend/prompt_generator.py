def build_counterecho_prompt(
    tweet: str,
    stance_vector: dict,
    counter_stance_vector: dict,
    user_category: str,
    user_camp: str,
    strategy: str,
    counter_narratives_str: str,
    stances_to_counter: dict
) -> str:
    """
    Builds a prompt for generating a counterecho message based on the provided inputs.
    
    :param tweet: The input tweet text.
    :param stance_vector: Dictionary representing the stance vector (e.g., {'LNG': 0, ...}).
    :param counter_stance_vector: Dictionary representing the counter stance vector.
    :param user_category: The user category ('open_minded', 'neutral', or 'closed_minded').
    :param user_camp: The user camp ('LEFT', 'RIGHT', or 'RU_CH').
    :param strategy: The strategy text for the user category.
    :param counter_narratives_str: Formatted string of relevant counter-narratives.
    :param stances_to_counter: Dictionary of stances we are countering.
    :return: The constructed prompt string.
    """
    
    # Format stance vectors as strings
    stance_str = "\n".join([f"{key}: {value}" for key, value in stance_vector.items() if value != 0])
    if not stance_str:
        stance_str = "No strong stances detected."
    
    # Format stances to counter
    stances_to_counter_str = "\n".join([
        f"- {key}: detected as {'pro' if value > 0 else 'anti'}, counter with {'anti' if value > 0 else 'pro'}"
        for key, value in stances_to_counter.items()
    ])
    if not stances_to_counter_str:
        stances_to_counter_str = "No stances to counter detected."
    
    # Updated link instruction
    link_instruction = (
        "If you believe the reply would benefit from credible evidence or factual support, "
        "you must include a relevant link to a trustworthy source. "
        "In such cases, explicitly search for accurate and current data to back up the message. "
        "Only include a link when it enhances credibility or context; otherwise, omit it."
    )
    
    # Construct the prompt
    prompt = f"""
You are a counterecho bot designed to break echo chambers by countering malign narratives. Your goal is to generate a reply message that subtly counters the narratives in the input tweet by promoting the opposite stances, focusing on US-LNG benefits while dismantling anti-US-LNG or pro-adversary views.

### Input Information:
- **Input Tweet:** {tweet}

- **Stances We Are Countering (if US_POLICY is anti, then that has to be countered. Others are optional):**
{stances_to_counter_str}

- **User Category:** {user_category}
- **User Camp:** {user_camp}

### Relevant Counter-Narratives (use these themes and language to craft your response):
{counter_narratives_str}

{strategy}

### Message Constraints:
- The message must be a direct reply to the tweet, starting with an appropriate hook as per the strategy.
- Keep it concise: 100-280 characters max, suitable for a tweet reply.
- Avoid complexity: Use simple language, no jargon unless specified in the strategy.
- Make it viral-friendly: Engaging, emotional, or curiosity-driven as per the strategy.
- Focus on countering only 1-3 of the stances listed above to keep it focused.
- Do not reveal that you are a bot or the countering intent directly.
- {link_instruction}

Search internet and add link if you deem the reply to require credibility or factual support.
CRITICAL: Output ONLY the tweet reply text itself (with or without link). No explanations, no reasoning, no markdown formatting, no headers, no labels like "Final Reply". Just the raw tweet reply message and nothing else.
"""
    
    return prompt
