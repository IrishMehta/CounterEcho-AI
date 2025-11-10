"""
Semantic Narrative Detection Module

This module detects narratives from tweet text using semantic embeddings
and cosine similarity instead of simple keyword matching.

Community-Camp Mapping:
- RU/CH CAMP: [C3, C7, C21, C12]
- LEFT CAMP: [C25, C10, C1, C5]
- RIGHT CAMP: [C2, C6]
"""

import json
import logging
import numpy as np
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class NarrativeDetector:
    """Semantic narrative detection using embeddings and cosine similarity."""
    
    def __init__(
        self, 
        narrative_definitions_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.35
    ):
        """
        Initialize semantic narrative detector.
        
        Args:
            narrative_definitions_path: Path to narratives.json
            model_name: HuggingFace sentence transformer model
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
                                 Lower = more permissive, Higher = more strict
        """
        logger.info(f"Loading narrative definitions from {narrative_definitions_path}")
        with open(narrative_definitions_path, 'r') as f:
            self.narrative_definitions = json.load(f)
        
        self.similarity_threshold = similarity_threshold
        logger.info(f"Similarity threshold set to: {similarity_threshold}")
        
        # Load sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Prepare narratives and compute embeddings
        self.narratives = self._prepare_narratives()
        logger.info(f"Loaded {len(self.narratives)} narrative patterns")
        
        self._compute_narrative_embeddings()
        logger.info("Narrative embeddings computed successfully")
    
    def _prepare_narratives(self) -> List[Dict]:
        """Prepare narrative data structure."""
        narratives = []
        for camp, data in self.narrative_definitions.items():
            if 'topics' in data:
                for topic, topic_data in data['topics'].items():
                    if 'narratives' in topic_data:
                        for label, pattern in topic_data['narratives'].items():
                            full_label = f"{camp.replace(' CAMP', '')} | {topic.replace(':', '')} | {label}"
                            
                            # Convert pattern to descriptive text for embedding
                            description = self._pattern_to_description(pattern, label)
                            
                            narratives.append({
                                'full_label': full_label,
                                'camp': camp,
                                'topic': topic,
                                'label': label,
                                'pattern': pattern,
                                'description': description
                            })
        return narratives
    
    def _pattern_to_description(self, pattern: str, label: str) -> str:
        """
        Convert keyword pattern to natural language description for better embeddings.
        
        Args:
            pattern: Keyword pattern from narratives.json
            label: Narrative label
            
        Returns:
            Natural language description
        """
        # Extract keywords and clean them
        cleaned = pattern.replace('{', '').replace('}', '')
        sections = [s.strip() for s in cleaned.split('+')]
        
        # Combine label with key concepts from each section
        # Take important keywords but avoid overly generic ones
        key_concepts = []
        for section in sections:
            keywords = [kw.strip() for kw in section.split(',')]
            # Filter out very short or generic terms
            important = [kw for kw in keywords if len(kw) > 2][:5]  # Take top 5
            key_concepts.extend(important)
        
        # Create a natural description
        description = f"{label}. Related to: {', '.join(key_concepts[:10])}"
        return description
    
    def _compute_narrative_embeddings(self):
        """Pre-compute embeddings for all narrative descriptions."""
        descriptions = [n['description'] for n in self.narratives]
        
        logger.info("Computing embeddings for narrative descriptions...")
        self.narrative_embeddings = self.model.encode(
            descriptions,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logger.info(f"Computed embeddings shape: {self.narrative_embeddings.shape}")
    
    def detect_narrative(
        self, 
        text: str,
        user_camp: Optional[str] = None,
        return_confidence: bool = False,
        top_k: int = 1
    ) -> Optional[str]:
        """
        Detect narrative from tweet text using semantic similarity.
        
        Args:
            text: Tweet text
            user_camp: User's camp ('RU_CH', 'LEFT', 'RIGHT') - if provided,
                      prioritizes narratives from this camp
            return_confidence: If True, return tuple of (label, confidence)
            top_k: Number of top matches to consider (default: 1)
            
        Returns:
            Narrative label or None (or tuple if return_confidence=True)
        """
        if not text or len(text.strip()) < 10:
            logger.debug("Text too short for narrative detection")
            return None if not return_confidence else (None, 0.0)
        
        # Encode the tweet
        tweet_embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Compute cosine similarities
        similarities = cosine_similarity(tweet_embedding, self.narrative_embeddings)[0]
        
        # Get global best match
        global_best_idx = np.argmax(similarities)
        global_best_similarity = similarities[global_best_idx]
        
        # If user camp is provided, try camp-based filtering
        if user_camp:
            # Normalize camp name to match narrative definitions
            camp_normalized = f"{user_camp} CAMP" if not user_camp.endswith(' CAMP') else user_camp
            
            # Find indices of narratives from user's camp
            camp_indices = [
                i for i, n in enumerate(self.narratives) 
                if camp_normalized in n['camp'] or n['camp'].replace(' CAMP', '') == user_camp
            ]
            
            if camp_indices:
                # Get best match within user's camp
                camp_similarities = similarities[camp_indices]
                best_camp_local_idx = np.argmax(camp_similarities)
                best_camp_idx = camp_indices[best_camp_local_idx]
                best_camp_similarity = similarities[best_camp_idx]
                
                # Use lower threshold for same-camp narratives (90% of normal threshold)
                camp_threshold = self.similarity_threshold * 0.9
                
                logger.debug(
                    f"Camp filtering active for '{user_camp}': "
                    f"Global best={global_best_similarity:.3f} ({self.narratives[global_best_idx]['camp']}), "
                    f"Camp best={best_camp_similarity:.3f} ({self.narratives[best_camp_idx]['camp']})"
                )
                
                # If camp match is reasonable, use it instead
                if best_camp_similarity >= camp_threshold:
                    best_idx = best_camp_idx
                    best_similarity = best_camp_similarity
                    logger.info(
                        f"Using camp-filtered narrative (similarity: {best_similarity:.3f})"
                    )
                else:
                    # Camp match not good enough, use global best
                    best_idx = global_best_idx
                    best_similarity = global_best_similarity
                    logger.debug(
                        f"Camp match below threshold ({best_camp_similarity:.3f} < {camp_threshold:.3f}), "
                        f"using global best"
                    )
            else:
                # No narratives found for this camp, use global best
                best_idx = global_best_idx
                best_similarity = global_best_similarity
                logger.warning(f"No narratives found for camp '{user_camp}', using global best")
        else:
            # No camp filtering, use global best
            best_idx = global_best_idx
            best_similarity = global_best_similarity
        
        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            best_narrative = self.narratives[best_idx]['full_label']
            logger.debug(
                f"Detected narrative: {best_narrative} "
                f"(similarity: {best_similarity:.3f})"
            )
            
            if return_confidence:
                return (best_narrative, float(best_similarity))
            return best_narrative
        else:
            logger.debug(
                f"No narrative detected. Best similarity: {best_similarity:.3f}, "
                f"threshold: {self.similarity_threshold}"
            )
            
            if return_confidence:
                return (None, float(best_similarity))
            return None
    
    def detect_narratives_batch(
        self, 
        texts: List[str],
        return_confidence: bool = False
    ) -> List[Optional[str]]:
        """
        Detect narratives for multiple tweets at once (more efficient).
        
        Args:
            texts: List of tweet texts
            return_confidence: If True, return list of tuples (label, confidence)
            
        Returns:
            List of narrative labels or None values
        """
        if not texts:
            return []
        
        # Filter out very short texts
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) >= 10]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return [None] * len(texts) if not return_confidence else [(None, 0.0)] * len(texts)
        
        # Encode all tweets at once
        logger.debug(f"Processing batch of {len(valid_texts)} tweets")
        tweet_embeddings = self.model.encode(
            valid_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Compute similarities for all tweets
        similarities = cosine_similarity(tweet_embeddings, self.narrative_embeddings)
        
        # Get best matches
        results = [None] * len(texts) if not return_confidence else [(None, 0.0)] * len(texts)
        
        for orig_idx, valid_idx in enumerate(valid_indices):
            sims = similarities[orig_idx]
            best_idx = np.argmax(sims)
            best_similarity = sims[best_idx]
            
            if best_similarity >= self.similarity_threshold:
                best_narrative = self.narratives[best_idx]['full_label']
                results[valid_idx] = (best_narrative, float(best_similarity)) if return_confidence else best_narrative
            else:
                results[valid_idx] = (None, float(best_similarity)) if return_confidence else None
        
        return results
    
    def get_top_k_narratives(
        self, 
        text: str, 
        k: int = 3
    ) -> List[tuple]:
        """
        Get top K most similar narratives for a tweet.
        
        Args:
            text: Tweet text
            k: Number of top matches to return
            
        Returns:
            List of (narrative_label, similarity_score) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        # Encode the tweet
        tweet_embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Compute cosine similarities
        similarities = cosine_similarity(tweet_embedding, self.narrative_embeddings)[0]
        
        # Get top K indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            narrative_label = self.narratives[idx]['full_label']
            similarity = float(similarities[idx])
            results.append((narrative_label, similarity))
        
        return results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    NARRATIVE_DEFINITIONS_PATH = "pipeline/narratives.json"
    
    # Initialize detector
    detector = NarrativeDetector(
        NARRATIVE_DEFINITIONS_PATH,
        similarity_threshold=0.35  # Adjust this based on your needs
    )
    
    # Test tweets
    test_tweets = [
        "China is securing its energy future through strategic partnerships with Russia",
        "Beijing continues to strengthen energy cooperation with Moscow",
        "The US imposed new tariffs on Chinese goods",
        "What a beautiful day today!",  # Should not match any narrative
    ]
    
    print("\n=== Single Tweet Detection ===")
    for tweet in test_tweets:
        narrative = detector.detect_narrative(tweet, return_confidence=True)
        print(f"\nTweet: {tweet}")
        print(f"Detected: {narrative}")
    
    print("\n=== Batch Detection ===")
    results = detector.detect_narratives_batch(test_tweets, return_confidence=True)
    for tweet, result in zip(test_tweets, results):
        print(f"\nTweet: {tweet}")
        print(f"Result: {result}")
    
    print("\n=== Top-3 Matches ===")
    tweet = "China buying Russian gas is good for energy security"
    top_k = detector.get_top_k_narratives(tweet, k=3)
    print(f"\nTweet: {tweet}")
    for i, (narrative, score) in enumerate(top_k, 1):
        print(f"{i}. {narrative} (similarity: {score:.3f})")