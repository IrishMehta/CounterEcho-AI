"""
Main Integration Script for Vulnerability Assessment and Strategy Mapping

This script:
1. Loads the retweet network graph and tweet data
2. Randomly selects a user and their tweet
3. Calculates user vulnerability score
4. Generates strategic counter-narrative using Groq API
5. Outputs the recommended response with strategy details

Community-Camp Mapping:
- RU/CH CAMP: [C3, C7, C21, C12]
- LEFT CAMP: [C25, C10, C1, C5]
- RIGHT CAMP: [C2, C6]
"""

import os
import sys

sys.path.append('./pipeline')

from pipeline.data_loader import load_graph_and_communities, load_tweets
from pipeline.user_selector import select_random_user_with_tweet
from pipeline.vulnerability_and_counter import VulnerabilityAssessor
from pipeline.narrative_detector import NarrativeDetector
from pipeline.strategy_mapping import StrategyMapper


def main():
    """Main execution function."""
    
    print("="*80)
    print("VULNERABILITY-BASED STRATEGIC COUNTER-NARRATIVE SYSTEM")
    print("="*80)
    print()
    
    GRAPH_PATH = "pipeline/all.gexf"
    COMMUNITY_PATH = "pipeline/community.csv"
    TWEETS_PATH = "pipeline/cleaned_dataset.csv"
    NARRATIVE_MAPPING_PATH = "pipeline/counter_narrative_mapping_combined.json"
    NARRATIVE_DEFINITIONS_PATH = "pipeline/narratives.json"
    
    G, community_df = load_graph_and_communities(GRAPH_PATH, COMMUNITY_PATH)
    tweets_df = load_tweets(TWEETS_PATH, sample_size=50000)
    
    print("\nInitializing components...")
    vulnerability_assessor = VulnerabilityAssessor(G, community_df)
    narrative_detector = NarrativeDetector(NARRATIVE_DEFINITIONS_PATH)
    
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("\nERROR: GROQ_API_KEY not found in environment variables.")
        print("   Set it with: export GROQ_API_KEY='your-key-here'")
        print("   Exiting...")
        return
    
    strategy_mapper = StrategyMapper(
        narrative_mapping_path=NARRATIVE_MAPPING_PATH,
        narrative_definitions_path=NARRATIVE_DEFINITIONS_PATH,
        groq_api_key=groq_api_key
    )
    
    print("\n" + "="*80)
    print("SELECTING RANDOM USER FROM NETWORK")
    print("="*80)
    
    user_id, community, camp, actual_tweet = select_random_user_with_tweet(
        community_df, tweets_df
    )
    
    print(f"\nUser Selected:")
    print(f"   User ID: {user_id}")
    print(f"   Community: {community}")
    print(f"   Camp: {camp}")
    print(f"\nActual Tweet:")
    print(f"   \"{actual_tweet}\"")
    
    print("\n" + "="*80)
    print("VULNERABILITY ASSESSMENT")
    print("="*80)
    
    vulnerability_score = vulnerability_assessor.get_vulnerability_score(user_id)
    
    print(f"\nVulnerability Score: {vulnerability_score:.3f}")
    print(f"   Interpretation: ", end="")
    if vulnerability_score > 0.7:
        print("HIGH - User is highly vulnerable to persuasion")
    elif vulnerability_score > 0.5:
        print("MODERATE - User has moderate vulnerability")
    else:
        print("LOW - User is entrenched in their position")
    
    print("\n" + "="*80)
    print("NARRATIVE DETECTION")
    print("="*80)
    
    detected_narrative = narrative_detector.detect_narrative(actual_tweet)
    
    if detected_narrative:
        print(f"\nDetected Narrative:")
        print(f"   {detected_narrative}")
    else:
        print("\nNo specific narrative detected from tweet text.")
        print(f"   Attempting to use camp-based default for camp: {camp}")
        default_narratives = {
            'RU_CH': 'RU_CH | pro-RUSSIA | Reliable and Vast Energy Supplier',
            'LEFT': 'LEFT | anti-Russia | Russia as an Unreliable and Threatening Energy Partner',
            'RIGHT': 'RIGHT | pro-US_USA_America | US Energy Dominance and Exports'
        }
        detected_narrative = default_narratives.get(camp)
        if detected_narrative:
            print(f"   Using default: {detected_narrative}")
        else:
            raise ValueError(f"No default narrative available for camp: {camp}")
    
    if detected_narrative:
        print("\n" + "="*80)
        print("STRATEGIC COUNTER-NARRATIVE GENERATION")
        print("="*80)
        
        result = strategy_mapper.map_strategy(
            user_id=user_id,
            tweet_text=actual_tweet,
            current_narrative=detected_narrative,
            vulnerability_score=vulnerability_score
        )
        
        print(f"\nStrategy Selected: {result['strategy'].upper()}")
        print(f"   Vulnerability: {result['vulnerability_score']:.3f}")
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
        else:
            if result['strategy'] == 'direct_counter':
                print(f"\nCounter-Narrative:")
                print(f"   {result['counter_narrative']}")
            elif result['strategy'] == 'adjacency_attack':
                print(f"\nAdjacent Narrative (Same Camp):")
                print(f"   {result['adjacent_narrative']}")
                print(f"   Contradiction Score: {result.get('contradiction_score', 'N/A')}")
            
            if 'generated_response' in result and 'message' in result['generated_response']:
                print("\n" + "="*80)
                print("RECOMMENDED RESPONSE")
                print("="*80)
                
                generated = result['generated_response']
                
                print(f"\nStrategic Intent:")
                if result['strategy'] == 'direct_counter':
                    print(f"   {generated.get('summary', 'Challenge the adversarial narrative directly')}")
                else:
                    print(f"   {generated.get('strategic_intent', 'Create cognitive dissonance')}")
                
                print(f"\nMessage to Post:")
                print(f"   {generated['message']}")
                
                print(f"\nMessage Characteristics:")
                print(f"   - Target Audience: {generated.get('target_audiences', 'N/A')}")
                print(f"   - Stance: {generated.get('stance', 'N/A')}")
                print(f"   - Label: {generated.get('label', 'N/A')}")
            else:
                raise ValueError("No message generated in response")
    else:
        raise ValueError("Cannot proceed without a detected narrative")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
