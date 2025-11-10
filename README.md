# CounterEcho-AI

A vulnerability-based strategic counter-narrative system that analyzes social network data to generate targeted counter-narratives based on user vulnerability scores.

## Overview

This system:
1. Loads retweet network graph and tweet data
2. Selects users and analyzes their tweets
3. Calculates user vulnerability scores based on network metrics
4. Detects narratives using semantic embeddings
5. Generates strategic counter-narratives using Groq API

### Community-Camp Mapping
- **RU/CH CAMP**: Communities C3, C7, C21, C12
- **LEFT CAMP**: Communities C25, C10, C1, C5
- **RIGHT CAMP**: Communities C2, C6

## Setup Instructions

### 1. Create Python Environment

Using conda (recommended):
```bash
conda create -n counterecho python=3.10
conda activate counterecho
```

Using venv:
```bash
python3 -m venv counterecho
source counterecho/bin/activate  # On Linux/Mac
# or
counterecho\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Key

The system requires a Groq API key for counter-narrative generation.

```bash
export GROQ_API_KEY='your-groq-api-key-here'
```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=your-groq-api-key-here
```

### 4. Verify Data Files

Ensure the following data files exist in the `pipeline/` directory:
- `all.gexf` - Network graph file
- `community.csv` - Community assignments
- `cleaned_dataset.csv` - Tweet data
- `counter_narrative_mapping_combined.json` - Narrative mappings
- `narratives.json` - Narrative definitions

## Running the System

```bash
python main.py
```

### Expected Output

The system will:
1. Load network and tweet data
2. Randomly select a user with a tweet
3. Display vulnerability assessment
4. Detect the narrative from the tweet
5. Generate a strategic counter-narrative
6. Display the recommended response

### Sample Output Structure

```
================================================================================
VULNERABILITY-BASED STRATEGIC COUNTER-NARRATIVE SYSTEM
================================================================================

SELECTING RANDOM USER FROM NETWORK
   User ID: 12345
   Community: C3
   Camp: RU/CH
   
   Actual Tweet:
      "Tweet text here..."

VULNERABILITY ASSESSMENT
   Vulnerability Score: 0.657
   Interpretation: MODERATE - User has moderate vulnerability

NARRATIVE DETECTION
   Detected Narrative: [narrative name]

STRATEGIC COUNTER-NARRATIVE GENERATION
   Strategy Selected: DIRECT_COUNTER
   Vulnerability: 0.657
   
   Counter-Narrative: [counter narrative]

RECOMMENDED RESPONSE
   Strategic Intent: [description]
   
   Message to Post: [generated message]
   
   Message Characteristics:
      - Target Audience: [audience]
      - Stance: [stance]
      - Label: [label]
```

## Project Structure

```
CounterEcho-AI/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── pipeline/
    ├── data_loader.py         # Data loading utilities
    ├── user_selector.py       # User selection logic
    ├── vulnerability_and_counter.py  # Vulnerability assessment
    ├── narrative_detector.py  # Semantic narrative detection
    ├── strategy_mapping.py    # Counter-narrative generation
    └── [data files]           # Data files
```

## Strategies

### Direct Counter Strategy
- Used for **vulnerable users** (high vulnerability score)
- Directly challenges the adversarial narrative
- Presents counter-narrative from opposing camp

### Adjacency Attack Strategy
- Used for **entrenched users** (low vulnerability score)
- Creates cognitive dissonance
- Uses contradictory narratives from the same camp

## Requirements

- Python 3.10+
- Groq API key
- Network graph data (GEXF format)
- Tweet dataset (CSV format)
- Narrative definition files (JSON format)
