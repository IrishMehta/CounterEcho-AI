"""
Strategic Counter-Narrative Mapping System

This module implements a two-strategy approach for counter-narrative generation:
1. Direct Counter: For vulnerable users (high vulnerability score)
2. Adjacency Attack: For entrenched users (low vulnerability score)

Uses Groq's Qwen3-32B reasoning model for message generation.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Constructs structured prompts for counter-narrative generation."""
    
    @staticmethod
    def build_direct_counter_prompt(
        adversarial_narrative: str,
        adversarial_camp: str,
        counter_narrative_labels: List[str],
        target_audience: str
    ) -> str:
        """
        Builds a direct counter-narrative prompt following Fig. 4 structure.
        
        Args:
            adversarial_narrative: The malign narrative to counter
            adversarial_camp: The camp/ideology of the adversarial narrative
            counter_narrative_labels: List of approved counter-narrative labels
            target_audience: Description of the target audience
            
        Returns:
            Structured prompt string for Groq API
        """
        counter_list = "\n".join([f"- {label}" for label in counter_narrative_labels])
        
        prompt = f"""Objective: Analyze the adversarial narrative labeled "{adversarial_narrative}" from the {adversarial_camp}. Construct a strategically sound, ethically grounded counter-narrative using approved patterned narratives from the following list:

{counter_list}

Instructions: Elevate in-group values, contrast with out-groups, and emphasize threats or protections to shared societal goods and personal values. Move beyond refutation—reframe the adversarial claim using symbolically coherent logic that aligns with the audience's moral commitments.

Output Format (provide as structured JSON):
{{
  "target_audiences": "{target_audience}",
  "stance": "pro/anti position relative to the subject",
  "label": "concise counter-narrative theme",
  "patterned_narrative": "symbolic logic expression",
  "summary": "purpose and strategic intent",
  "message": "the actual counter-narrative message to post"
}}

Generate a compelling counter-narrative that will resonate with the target audience and effectively challenge the adversarial narrative."""
        
        return prompt
    
    @staticmethod
    def build_adjacency_attack_prompt(
        current_narrative: str,
        current_camp: str,
        adjacent_narrative: str,
        target_audience: str
    ) -> str:
        """
        Builds an adjacency attack prompt to expose internal contradictions.
        
        Args:
            current_narrative: The narrative being pushed
            current_camp: The camp/ideology of the current narrative
            adjacent_narrative: Contradictory narrative from same camp
            target_audience: Description of the target audience
            
        Returns:
            Structured prompt string for Groq API
        """
        prompt = f"""Objective: Identify an internal contradiction within the {current_camp} by leveraging a related but contradictory narrative from their own ideological camp.

Current Narrative: {current_narrative}
Adjacent Contradictory Narrative: {adjacent_narrative}

Instructions: 
- Highlight the tension between these two narratives from the same camp
- Use Socratic questioning to expose the inconsistency
- Avoid direct opposition - instead, present this as a "fellow traveler's concern"
- Frame it as protecting the in-group's true values from being undermined
- Show genuine curiosity about how both positions can be reconciled

Output Format (provide as structured JSON with ALL fields filled):
{{
  "target_audiences": "description of who this targets",
  "stance": "your apparent alignment with in-group values",
  "adjacent_narrative": "name of the contradictory narrative",
  "cognitive_dissonance_frame": "REQUIRED: explain specifically how these two narratives conflict and create tension",
  "strategic_intent": "REQUIRED: explain how this approach sows doubt without triggering defensive reactance",
  "message": "the actual message to post (use questioning, not accusatory tone)"
}}

IMPORTANT: Fill in ALL fields with substantive content. Do not use placeholder text. Generate a message that creates cognitive dissonance by highlighting this internal contradiction."""
        
        return prompt


class NarrativeEmbedder:
    """Handles embedding-based narrative similarity calculations."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        logger.info("Embedding model loaded successfully")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text."""
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = self.model.encode(text, convert_to_numpy=True)
        return self.embeddings_cache[text]
    
    def calculate_contradiction_score(self, narrative1: str, narrative2: str) -> float:
        """
        Calculate how contradictory two narratives are.
        
        Lower cosine similarity = higher contradiction (more different semantically)
        
        Args:
            narrative1: First narrative text
            narrative2: Second narrative text
            
        Returns:
            Contradiction score (0-1, higher = more contradictory)
        """
        emb1 = self.get_embedding(narrative1)
        emb2 = self.get_embedding(narrative2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Convert to contradiction score (1 - similarity)
        contradiction = 1 - similarity
        
        return float(contradiction)
    
    def select_most_contradictory(
        self, 
        current_narrative: str, 
        candidate_narratives: List[str]
    ) -> Tuple[str, float]:
        """
        Select the most contradictory narrative from candidates.
        
        Args:
            current_narrative: The narrative to counter
            candidate_narratives: List of potential counter-narratives
            
        Returns:
            Tuple of (most_contradictory_narrative, contradiction_score)
        """
        if not candidate_narratives:
            return None, 0.0
        
        scores = [
            (narrative, self.calculate_contradiction_score(current_narrative, narrative))
            for narrative in candidate_narratives
        ]
        
        # Sort by contradiction score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[0]


class MessageGenerator:
    """Handles interaction with Groq API for message generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (if None, uses GROQ_API_KEY env variable)
        """
        logger.info("Initializing Groq API client")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = Groq()
        logger.info("Groq client initialized successfully")
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.32,
        max_tokens: int = 4096
    ) -> Dict:
        """
        Generate a response using Groq's Qwen3-32B model.
        
        Args:
            prompt: The structured prompt
            temperature: Sampling temperature (0.32 for more focused responses)
            max_tokens: Maximum completion tokens
            
        Returns:
            Parsed JSON response from the model
        """
        try:
            logger.debug(f"Generating message with temperature={temperature}, max_tokens={max_tokens}")
            completion = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=0.95,
                reasoning_effort="default",
                stream=False,
                stop=None
            )
            
            response_text = completion.choices[0].message.content
            logger.debug(f"Received response from Groq API ({len(response_text)} chars)")
            
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    logger.info("Successfully parsed JSON response from Groq")
                    return result
                else:
                    logger.warning("No JSON structure found in response, returning raw text")
                    return {"raw_response": response_text}
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON: {je}")
                return {"raw_response": response_text}
                
        except Exception as e:
            logger.error(f"Error generating message: {e}")
            return {"error": str(e)}


class StrategyMapper:
    """
    Main orchestrator for strategic counter-narrative mapping.
    
    Selects between direct counter and adjacency attack strategies based on
    vulnerability scores and generates appropriate counter-messages.
    """
    
    def __init__(
        self, 
        narrative_mapping_path: str,
        narrative_definitions_path: str,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize the strategy mapper.
        
        Args:
            narrative_mapping_path: Path to counter_narrative_mapping_combined.json
            narrative_definitions_path: Path to narratives.json
            groq_api_key: Optional Groq API key
        """
        logger.info("Initializing Strategy Mapper...")
        
        logger.info(f"Loading narrative mappings from {narrative_mapping_path}")
        with open(narrative_mapping_path, 'r') as f:
            self.narrative_mapping = json.load(f)
        
        logger.info(f"Loading narrative definitions from {narrative_definitions_path}")
        with open(narrative_definitions_path, 'r') as f:
            self.narrative_definitions = json.load(f)
        
        logger.info("Initializing components...")
        self.prompt_builder = PromptBuilder()
        self.embedder = NarrativeEmbedder()
        self.message_generator = MessageGenerator(groq_api_key)
        
        logger.info("Strategy Mapper initialized successfully")
    
    def select_strategy(self, vulnerability_score: float, threshold: float = 0.6) -> str:
        """
        Select strategy based on vulnerability score.
        
        Args:
            vulnerability_score: User's vulnerability score (0-1)
            threshold: Threshold for strategy selection
            
        Returns:
            'direct_counter' or 'adjacency_attack'
        """
        return 'direct_counter' if vulnerability_score > threshold else 'adjacency_attack'
    
    def _select_best_counter(
        self, 
        current_narrative: str, 
        mapping_entry: Dict
    ) -> Optional[str]:
        """
        Select the best counter-narrative based on data availability.
        
        Priority: right_camp_counter > ru_ch_camp_counter > left_self_counter
        
        Args:
            current_narrative: The narrative to counter
            mapping_entry: Entry from counter_narrative_mapping_combined.json
            
        Returns:
            Selected counter-narrative label or None
        """
        # Priority order based on availability
        for key in ['right_camp_counter', 'ru_ch_camp_counter', 'left_self_counter']:
            if key in mapping_entry and mapping_entry[key] not in [None, "None", ""]:
                return mapping_entry[key]
        
        return None
    
    def _extract_camp(self, narrative_label: str) -> str:
        """Extract camp name from narrative label (e.g., 'LEFT | pro-China | ...' -> 'LEFT')"""
        parts = narrative_label.split('|')
        return parts[0].strip() if parts else "Unknown"
    
    def _get_narrative_description(self, narrative_label: str, include_pattern: bool = True) -> str:
        """
        Get the full description/pattern of a narrative from narratives.json.
        
        Args:
            narrative_label: The narrative label
            include_pattern: Whether to include the full symbolic pattern (default: True)
            
        Returns:
            Narrative description or the label itself if not found
        """
        # Search through the nested structure
        for camp, camp_data in self.narrative_definitions.items():
            if 'topics' in camp_data:
                for topic, topic_data in camp_data['topics'].items():
                    if 'narratives' in topic_data:
                        for narr_label, narr_desc in topic_data['narratives'].items():
                            if narr_label in narrative_label or narrative_label in narr_label:
                                if include_pattern:
                                    return f"{narr_label}: {narr_desc}"
                                else:
                                    return narr_label
        
        return narrative_label
    
    def map_strategy(
        self, 
        user_id: str,
        tweet_text: str,
        current_narrative: str,
        vulnerability_score: float,
        threshold: float = 0.6
    ) -> Dict:
        """
        Main orchestration function for strategy mapping.
        
        Args:
            user_id: User identifier
            tweet_text: The tweet text
            current_narrative: Detected narrative label
            vulnerability_score: User's vulnerability score
            threshold: Threshold for strategy selection
            
        Returns:
            Dictionary containing strategy, prompt, and generated message
        """
        strategy = self.select_strategy(vulnerability_score, threshold)
        
        logger.info(f"Processing strategy for user {user_id}")
        logger.info(f"Selected strategy: {strategy} (vulnerability: {vulnerability_score:.3f})")
        
        result = {
            "user_id": user_id,
            "tweet_text": tweet_text,
            "current_narrative": current_narrative,
            "vulnerability_score": round(vulnerability_score, 3),
            "strategy": strategy,
        }
        
        if current_narrative not in self.narrative_mapping:
            logger.error(f"No mapping found for narrative: {current_narrative}")
            result["error"] = f"No mapping found for narrative: {current_narrative}"
            return result
        
        mapping_entry = self.narrative_mapping[current_narrative]
        current_camp = self._extract_camp(current_narrative)
        
        if strategy == 'direct_counter':
            logger.info("Executing direct counter strategy")
            counter_narrative = self._select_best_counter(current_narrative, mapping_entry)
            
            if not counter_narrative:
                logger.error("No counter-narrative available")
                result["error"] = "No counter-narrative available"
                return result
            
            logger.info(f"Selected counter-narrative: {counter_narrative}")
            result["counter_narrative"] = counter_narrative
            
            current_desc = self._get_narrative_description(current_narrative)
            counter_desc = self._get_narrative_description(counter_narrative)
            
            prompt = self.prompt_builder.build_direct_counter_prompt(
                adversarial_narrative=current_desc,
                adversarial_camp=current_camp,
                counter_narrative_labels=[counter_desc],
                target_audience=f"Users engaging with {current_camp} narratives"
            )
            
            result["prompt"] = prompt
            
            logger.info("Generating direct counter message via Groq API")
            response = self.message_generator.generate(prompt)
            result["generated_response"] = response
            
        elif strategy == 'adjacency_attack':
            logger.info("Executing adjacency attack strategy")
            self_counter_key = f"{current_camp.lower()}_self_counter"
            
            if self_counter_key not in mapping_entry or mapping_entry[self_counter_key] in [None, "None", ""]:
                logger.error(f"No self-counter available for adjacency attack (key: {self_counter_key})")
                result["error"] = f"No self-counter available for adjacency attack"
                return result
            
            adjacent_narrative = mapping_entry[self_counter_key]
            logger.info(f"Selected adjacent narrative: {adjacent_narrative}")
            result["adjacent_narrative"] = adjacent_narrative
            
            current_desc = self._get_narrative_description(current_narrative)
            adjacent_desc = self._get_narrative_description(adjacent_narrative)
            
            contradiction_score = self.embedder.calculate_contradiction_score(
                current_desc, 
                adjacent_desc
            )
            logger.debug(f"Contradiction score: {contradiction_score:.3f}")
            result["contradiction_score"] = round(contradiction_score, 3)
            
            # Build prompt
            prompt = self.prompt_builder.build_adjacency_attack_prompt(
                current_narrative=current_desc,
                current_camp=current_camp,
                adjacent_narrative=adjacent_desc,
                target_audience=f"Users engaging with {current_camp} narratives"
            )
            
            result["prompt"] = prompt
            
            # Generate message
            print(f"Generating adjacency attack for {current_narrative}...")
            response = self.message_generator.generate(prompt)
            result["generated_response"] = response
        
        return result


# --- MAIN EXECUTION ---

def main():
    """Example usage of the StrategyMapper."""
    
    # Initialize the mapper
    mapper = StrategyMapper(
        narrative_mapping_path="counter_narrative_mapping_combined.json",
        narrative_definitions_path="narratives.json",
        groq_api_key=None  # Uses GROQ_API_KEY from environment
    )
    
    # Example 1: Direct Counter (vulnerable user)
    print("\n" + "="*80)
    print("EXAMPLE 1: DIRECT COUNTER STRATEGY")
    print("="*80)
    
    result1 = mapper.map_strategy(
        user_id="user_12345",
        tweet_text="Europe must eliminate their reliance on Russian fossil fuels immediately.",
        current_narrative="LEFT | pro-Russia | Russia as a Stable and Reliable (Claimed) Energy Partner",
        vulnerability_score=0.75  # High vulnerability -> direct counter
    )
    
    print(json.dumps(result1, indent=2))
    
    # Example 2: Adjacency Attack (entrenched user)
    print("\n" + "="*80)
    print("EXAMPLE 2: ADJACENCY ATTACK STRATEGY")
    print("="*80)
    
    result2 = mapper.map_strategy(
        user_id="user_67890",
        tweet_text="The US energy dominance is reshaping global markets.",
        current_narrative="LEFT | pro-US_USA_America | largest net exporter of natural gas",
        vulnerability_score=0.3  # Low vulnerability -> adjacency attack
    )
    
    print(json.dumps(result2, indent=2))
    
    # Save results
    results = {
        "example_1_direct_counter": result1,
        "example_2_adjacency_attack": result2
    }
    
    with open("strategy_mapping_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Results saved to strategy_mapping_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
