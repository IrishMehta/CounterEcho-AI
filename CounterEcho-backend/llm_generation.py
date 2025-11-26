"""
LLM client for generating counter messages using Groq compound model.
"""
import os
from typing import Optional
from groq import Groq


# System message to enforce output format
SYSTEM_MESSAGE = """You are a tweet reply generator. Your ONLY output must be the tweet reply text itself.
Do NOT include:
- Explanations or reasoning
- Markdown formatting like **bold** or headers
- Labels like "Final Reply" or "Tweet-ready"
- Any meta-commentary about your approach
- Quotes around the reply

Just output the raw tweet reply text, nothing else."""


def generate_counter_message(prompt: str, temperature: float = 0.46, max_tokens: int = 1024) -> str:
    """
    Generate a counter message using Groq's compound model with web search capability.
    
    :param prompt: The full prompt for generating the counter message.
    :param temperature: Temperature for generation (default 0.46).
    :param max_tokens: Maximum tokens for completion (default 751).
    :return: Generated counter message text.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    completion = client.chat.completions.create(
        model="groq/compound",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
        compound_custom={"tools": {"enabled_tools": ["web_search", "visit_website"]}}
    )
    
    return completion.choices[0].message.content


# def generate_counter_message_simple(prompt: str, model: str = "llama-3.3-70b-versatile", 
#                                      temperature: float = 0.7, max_tokens: int = 500) -> str:
#     """
#     Generate a counter message using a standard Groq model (without compound features).
#     Fallback option if compound model is not available.
    
#     :param prompt: The full prompt for generating the counter message.
#     :param model: Model to use (default llama-3.3-70b-versatile).
#     :param temperature: Temperature for generation (default 0.7).
#     :param max_tokens: Maximum tokens for completion (default 500).
#     :return: Generated counter message text.
#     """
#     client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
#     completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=1,
#         stream=False,
#         stop=None,
#     )
    
#     return completion.choices[0].message.content
