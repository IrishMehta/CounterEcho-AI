"""
Narrative Detection Module

This module detects narratives from tweet text based on keyword matching.

Community-Camp Mapping:
- RU/CH CAMP: [C3, C7, C21, C12]
- LEFT CAMP: [C25, C10, C1, C5]
- RIGHT CAMP: [C2, C6]
"""

import json
import logging
from typing import Optional, List, Dict, Set

logger = logging.getLogger(__name__)


class NarrativeDetector:
    """Simple narrative detection based on keyword matching."""
    
    def __init__(self, narrative_definitions_path: str):
        """
        Initialize narrative detector.
        
        Args:
            narrative_definitions_path: Path to narratives.json
        """
        logger.info(f"Loading narrative definitions from {narrative_definitions_path}")
        with open(narrative_definitions_path, 'r') as f:
            self.narrative_definitions = json.load(f)
        
        self.flat_narratives = self._flatten_narratives()
        logger.info(f"Loaded {len(self.flat_narratives)} narrative patterns")
    
    def _flatten_narratives(self) -> List[Dict]:
        """Flatten narrative structure for easier searching."""
        flat_list = []
        for camp, data in self.narrative_definitions.items():
            if 'topics' in data:
                for topic, topic_data in data['topics'].items():
                    if 'narratives' in topic_data:
                        for label, pattern in topic_data['narratives'].items():
                            keywords = self._extract_keywords(pattern)
                            full_label = f"{camp.replace(' CAMP', '')} | {topic.replace(':', '')} | {label}"
                            flat_list.append({
                                'full_label': full_label,
                                'camp': camp,
                                'keywords': keywords,
                                'pattern': pattern
                            })
        return flat_list
    
    def _extract_keywords(self, pattern: str) -> Set[str]:
        """Extract keywords from narrative pattern."""
        cleaned = pattern.replace('{', ' ').replace('}', ' ').replace('+', ' ')
        keywords = [kw.strip().lower() for kw in cleaned.split(',')]
        return set(keywords)
    
    def detect_narrative(self, text: str, min_matches: int = 2) -> Optional[str]:
        """
        Detect narrative from tweet text.
        
        Args:
            text: Tweet text
            min_matches: Minimum keyword matches required (default: 2)
            
        Returns:
            Narrative label or None
        """
        text_lower = text.lower()
        
        best_match = None
        best_score = 0
        
        for narrative in self.flat_narratives:
            matches = sum(1 for kw in narrative['keywords'] if kw and len(kw) > 2 and kw in text_lower)
            
            if matches > best_score and matches >= min_matches:
                best_score = matches
                best_match = narrative['full_label']
        
        if best_match:
            logger.debug(f"Detected narrative with {best_score} keyword matches: {best_match}")
        else:
            logger.debug(f"No narrative detected (best score: {best_score}, min required: {min_matches})")
        
        return best_match
