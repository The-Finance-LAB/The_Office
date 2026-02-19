# The Office — Multi-Agent Financial Analyst Roundtable

An interactive AI-powered analyst roundtable that simulates a Wall Street conference room discussion about Apple's financial performance. Users ask a question and watch as 7 AI agents — each with a unique personality and fiscal year expertise — debate the data in real time.

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────┐
│   GitHub Pages      │  HTTP   │   Render (Python)       │
│   Static Frontend   │ ──────► │   Flask API Backend     │
│   (HTML/CSS/JS)     │         │                         │
│                     │ ◄────── │   LangChain Agents      │
│   Dark Glass UI     │  JSON   │   Gemini LLM            │
│   Polling + Queue   │         │   PyMuPDF + ChromaDB    │
└─────────────────────┘         └─────────────────────────┘
```

### Backend (Python Flask)
- **LangChain** multi-agent orchestration
- **Google Gemini** (gemini-2.0-flash) for dialogue generation
- **PyMuPDF** for PDF table and text extraction
- **ChromaDB** for vector similarity search
- **SQLite** for structured financial data storage
- Preserves the original Colab notebook architecture

### Frontend (Static HTML/CSS/JS)
- Dark Glass Trading Floor design
- Animated chat bubbles with agent-colored glow effects
- 3-second delays between messages with typing indicators
- Frosted glass (glassmorphism) UI components
- Responsive sidebar with participant profiles
- No build step required — pure vanilla JS

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/The-Finance-LAB/The_Office.git
cd The_Office
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Google API key
```

Required keys in `.env`:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 4. Add PDF Filings

Place your Apple 10-K SEC filing PDFs in the `backend/filings/` directory. The files should be named with fiscal year identifiers (e.g., `aapl_10k_fy2020.pdf`).

### 5. Run Locally

```bash
# From the backend directory
python app.py
```

The API will start on `http://localhost:5000`.

Then open `frontend/index.html` in your browser, or serve it:
```bash
cd ../frontend
python -m http.server 8080
```

Visit `http://localhost:8080` and update `frontend/js/config.js` to point to `http://localhost:5000`.

## Deploy to Render

### Backend Deployment

1. Push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Configure:
   - **Name**: `the-office-api`
   - **Root Directory**: `backend`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
6. Add environment variable:
   - `GOOGLE_API_KEY` = your Gemini API key
7. Deploy

Your API will be live at `https://the-office-api.onrender.com` (or similar).

### Frontend Deployment (GitHub Pages)

1. Update `frontend/js/config.js`:
   ```js
   API_BASE_URL: "https://the-office-api.onrender.com"
   ```
2. Push to GitHub
3. Go to **Settings → Pages** in your GitHub repo
4. Set Source to **Deploy from a branch**
5. Select `main` branch, `/frontend` folder (or use the `gh-pages` branch method below)

## Embed in Your Website

Once deployed, use this iframe code:

```html
<iframe
  src="https://the-finance-lab.github.io/The_Office/"
  width="100%"
  height="800"
  style="border: none; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
  allow="autoplay"
  loading="lazy"
  title="The Apple Brief — Analyst Roundtable">
</iframe>
```

## The Analysts

| Agent | Year | Personality | Color |
|-------|------|-------------|-------|
| **Chief Analyst** | All | Lead — authoritative, sharp | Gold |
| **Marcus** | FY2020 | The Veteran — dry humor, historical context | Steel Blue |
| **Priya** | FY2021 | Growth Evangelist — energetic, metaphors | Rose |
| **James** | FY2022 | The Contrarian — clinical, margin analysis | Teal |
| **Sofia** | FY2023 | Macro Thinker — geopolitical, FX lenses | Amber |
| **Derek** | FY2024 | The Quant — CAGRs, regressions, models | Ocean Blue |
| **Anika** | FY2025 | The Closer — forward-looking, synthesizer | Purple |

## Project Structure

```
The_Office/
├── backend/
│   ├── app.py              # Flask API + LangChain agents
│   ├── requirements.txt    # Python dependencies
│   ├── Procfile            # Render/Heroku start command
│   ├── runtime.txt         # Python version
│   ├── .env.example        # Environment variable template
│   └── filings/            # Apple 10-K PDF files
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── css/
│   │   └── styles.css      # Dark Glass Trading Floor theme
│   └── js/
│       ├── config.js       # API URL + timing configuration
│       ├── agents.js       # Analyst definitions
│       ├── components.js   # UI rendering functions
│       └── app.js          # Main application logic
├── render.yaml             # Render deployment config
├── .gitignore
└── README.md
```

## License

MIT
