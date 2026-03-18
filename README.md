# Cold Calling Leads — Solo-Owner Local Businesses

A **Streamlit** web app that scrapes **Google Maps** for local businesses and uses **Ollama** (qwen2.5:7b) locally to detect solo-owned businesses—ideal for cold calling leads. No paid APIs required.

## Features

- **Sidebar**: City (default Dallas), business type dropdown (plumber, HVAC, electrician, locksmith, garage door, roofer, dentist, chiropractor), max pages (1–10), **Start Scraping** button
- **Live UI**: Progress bar, status messages, real-time table preview as results come in
- **Results table**: business_name, address, phone, website, owner_name, confidence_score, num_reviews
- **Download**: “Download Solo Owners CSV” for rows where solo owner confidence > 0.7
- **Owner detection**: FREE local Ollama analyzes review text to infer a single dominant owner (e.g. “Josh did amazing work”) and outputs JSON with `owner_name`, `solo`, `confidence`, `reason`

## Requirements

- **Python 3.10+**
- **Ollama** installed and running with **qwen2.5:7b**
- **Playwright** Chromium (installed via `playwright install chromium`)

## Setup

### 1. Clone or copy this folder

```bash
cd maps-leads-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure Ollama

- Install [Ollama](https://ollama.ai) and start it (e.g. run `ollama serve` or use the desktop app).
- Pull the model:

```bash
ollama pull qwen2.5:7b
```

- Optional: copy `.env.example` to `.env` and set `OLLAMA_BASE_URL` or `OLLAMA_MODEL` if needed.

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Usage

1. Set **City** (e.g. Dallas) and **Business type** (e.g. plumber).
2. Choose **Max result pages** (more pages = more businesses, longer run).
3. Click **Start Scraping**.
4. Watch progress and the live table; when finished, use **Download Solo Owners CSV** to export leads with solo owner confidence > 0.7.

## Project layout

```
maps-leads-app/
├── app.py                 # Main Streamlit app + scraping + Ollama logic
├── requirements.txt       # Python dependencies
├── .env.example           # Optional env vars (OLLAMA_BASE_URL, OLLAMA_MODEL)
├── README.md              # This file
└── .streamlit/
    └── config.toml        # Streamlit theme and server config
```

## Notes

- **Google Maps**: Scraping is done with Playwright in headless mode. If the layout changes, selectors in `app.py` may need updating.
- **Ollama**: Must be running locally; the app uses `langchain-ollama` to call `qwen2.5:7b` for owner detection from review text.
- **Rate / ToS**: Use responsibly; consider Google’s terms of service and rate limits when scraping.

## License

Use at your own risk. For educational and internal lead generation only.
