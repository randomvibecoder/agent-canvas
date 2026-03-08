# Agent Canvas

AI-powered canvas for creating matplotlib visualizations using natural language.

## Features

- **Natural Language to Charts**: Describe what you want to visualize, and AI creates it
- **File Upload**: Upload `.txt` files with data and ask AI to graph it
- **1920x1080 Canvas**: High-resolution canvas for crisp visualizations
- **Markdown Support**: AI responses render beautifully with markdown
- **Dark Mode**: Beautiful dark theme UI
- **Real-time Sync**: Canvas updates in real-time across all connected clients

## Running

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:6473

## Usage

1. Visit the homepage and click "Try Now"
2. Ask AI to create visualizations like:
   - "Create a bar chart of [1, 2, 3, 4]"
   - "Show my electric bills over 10 years"
3. Upload a `.txt` file with data and ask AI to visualize it

## Tech Stack

- **Backend**: FastAPI (Python)
- **Canvas**: Fabric.js
- **LLM**: Nano GPT API
- **Visualization**: Matplotlib
