# Earnings Analyzer Backend

AI Agent Swarm for Financial Analysis - REST API Backend

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your AI Mode

**Option A: Claude API (Best Quality)**
```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

**Option B: Ollama (Free)**
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.1
```

**Option C: Demo Mode (Samples Only)**
```bash
# No setup needed - just run!
```

### 3. Run Backend

```bash
uvicorn api:app --reload --port 8000
```

Backend will be available at: **http://localhost:8000**

## 📚 API Endpoints

### GET `/`
Health check and API information

### GET `/api/health`
Detailed health check with component status

### POST `/api/analyze`
Analyze earnings call from text

**Request:**
```json
{
  "text": "Q3 2025 Earnings Call - Company Inc. CEO: Revenue grew 25%..."
}
```

**Response:**
```json
{
  "timestamp": "2025-02-07T15:30:00",
  "consensus": {
    "overall_score": 7.4,
    "verdict": "MET EXPECTATIONS - HOLD/BUY",
    "confidence": "Medium",
    "recommendation": "Solid results. Hold position..."
  },
  "detailed_analysis": {
    "revenue": {
      "score": 8.2,
      "verdict": "STRONG",
      "key_metrics": {...},
      "highlights": [...],
      "concerns": [...]
    },
    "profitability": {...},
    "management": {...}
  }
}
```

### POST `/api/analyze-pdf`
Analyze earnings call from PDF upload

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze-pdf \
  -F "file=@earnings_q3.pdf"
```

### GET `/api/samples`
Get list of pre-loaded sample analyses

**Response:**
```json
{
  "samples": [
    {
      "id": "techcorp_q3_2025",
      "company": "TechCorp Inc - Q3 2025",
      "overall_score": 7.4
    },
    {
      "id": "tesla_q3_2024",
      "company": "Tesla Inc - Q3 2024",
      "overall_score": 6.8
    }
  ],
  "count": 2
}
```

### GET `/api/sample/{sample_id}`
Get detailed analysis for specific sample

**Example:**
```bash
curl http://localhost:8000/api/sample/techcorp_q3_2025
```

## 🧪 Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/

# Analyze text
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Q3 Earnings Call - Revenue grew 25% to $2.8B..."}'

# Get samples
curl http://localhost:8000/api/samples

# Get specific sample
curl http://localhost:8000/api/sample/tesla_q3_2024
```

### Test with Python

```python
import requests

# Analyze earnings
response = requests.post(
    'http://localhost:8000/api/analyze',
    json={
        'text': 'Q3 2025 Earnings... Revenue grew 23%...'
    }
)

results = response.json()
print(f"Overall Score: {results['consensus']['overall_score']}/10")
print(f"Verdict: {results['consensus']['verdict']}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional - Claude API
export ANTHROPIC_API_KEY='sk-ant-your-key'

# Optional - Custom port
export PORT=8000
```

### Using .env file

Create `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
PORT=8000
```

Then in api.py:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 📊 API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger documentation

- Try endpoints directly in browser
- See request/response schemas
- Test with sample data

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is already in use:
lsof -i :8000

# Kill existing process:
kill -9 <PID>

# Or use different port:
uvicorn api:app --port 8001
```

### CORS errors
```bash
# Make sure CORS middleware is enabled in api.py
# Check frontend is calling correct URL
```

### Import errors
```bash
# Make sure earnings_analyzer.py is in same directory
ls earnings_analyzer.py

# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall
```

### Ollama not working
```bash
# Check Ollama is running:
ollama list

# Restart Ollama:
ollama serve

# Test Ollama:
ollama run llama3.1 "Hello"
```

## 📦 Project Structure

```
backend/
├── api.py                  ← FastAPI backend (this file)
├── earnings_analyzer.py    ← AI agent swarm
├── requirements.txt        ← Dependencies
├── .env                    ← Environment variables (optional)
├── README.md              ← This file
└── test_api.py            ← Test script (optional)
```

## 🚀 Deployment

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### Deploy to Render

1. Connect GitHub repo
2. Select "Web Service"
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## 💡 Development Tips

- Use `--reload` flag during development (auto-restart on code changes)
- Check `/docs` endpoint for interactive API testing
- Use `/api/health` to verify all components working
- Test with samples first before using real AI

## ⚡ Performance

- Average response time: 3-5 seconds (Claude API)
- Average response time: 5-8 seconds (Ollama)
- Sample loading: <100ms (instant)
- Concurrent requests: Supported via async
