# CounterEcho-AI

CounterEcho-AI is a vulnerability-based strategic counter-narrative system designed to analyze social network data and generate targeted counter-narratives. It leverages network metrics for user vulnerability assessment and Large Language Models (LLMs) for content generation.

## Project Structure

```
CounterEcho-AI/
├── ANCO-HITS/                 # User ranking and categorization algorithms
├── CounterEcho-backend/       # FastAPI backend for analysis & generation
├── CounterEcho-frontend/      # Flask frontend web interface
├── Stance-Detection/          # Stance detection modules
├── Virality-Analysis/         # Virality analysis tools
├── data/                      # Network graphs, datasets, and narrative mappings
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Prerequisites

- Python 3.10+
- [Groq API Key](https://groq.com/)
- [Google Gemini API Key](https://ai.google.dev/)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CounterEcho-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install flask requests
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Running the Application

The system consists of a backend API and a frontend UI. You need to run both in separate terminals.

### 1. Start the Backend Server

Navigate to the backend directory and start the FastAPI server:

```bash
cd CounterEcho-backend
uvicorn main:app --reload --port 8000
```

The backend API will be available at `http://localhost:8000`.

### 2. Start the Frontend Server

Navigate to the frontend directory and start the Flask UI:

```bash
cd CounterEcho-frontend
python server.py
```

The web interface will be available at `http://localhost:5005`.

## Usage

1.  Open your browser and go to `http://localhost:5005`.
2.  Enter a **User ID** and the **Tweet Content** you want to counter.
3.  The system will:
    - Analyze the user's vulnerability and stance.
    - Select an appropriate counter-strategy.
    - Generate a counter-narrative message.



