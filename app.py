from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
import requests # Make sure to run: pip install requests
from datetime import datetime
from pydantic import BaseModel

# Import your Bot Logic
from Kbot import ResearchBot, GeminiProvider, DuckDuckGoSearchProvider

app = FastAPI()

# --- 1. CONFIGURATION ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# GitHub Config
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
GITHUB_REPO = "Shamiyam/Research-Assistant" # Your specific repo

# --- 2. DATA MODELS ---
class FeedbackModel(BaseModel):
    rating: int
    comment: str

# --- 3. ROUTES ---

# A. Serve the Frontend
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    # Ensure index.html exists in the same folder
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# B. The Research Bot API
@app.get("/api/ask")
async def ask(question: str):
    llm = GeminiProvider()
    search = DuckDuckGoSearchProvider()
    bot = ResearchBot(llm, search)
    
    # This returns the full dictionary (answer + sources)
    result = await bot.research(question)
    return result 

# C. The Feedback API (Connects to GitHub)
@app.post("/api/feedback")
def submit_feedback(feedback: FeedbackModel): 
    # NOTE: I removed 'async' above because 'requests' is synchronous. 
    # FastAPI will handle this better in a threadpool.
    
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN is missing on the server.")
        return {"message": "Server configuration error"}, 500

    # Create the Issue Title and Body
    title = f"Feedback: User Rating {feedback.rating}/5"
    body = f"""
### User Feedback Report
**Rating:** {feedback.rating} stars
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Comment:**
{feedback.comment}

*Submitted via ResearchBot Interface*
    """

    # Send to GitHub API
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "title": title,
        "body": body,
        "labels": ["user-feedback", "enhancement"] 
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 201:
            return {"message": "Feedback added to GitHub Issues!"}
        else:
            print(f"GitHub Error: {response.text}")
            return {"message": "Failed to save to GitHub"}, 500
    except Exception as e:
        print(f"Network Error: {e}")
        return {"message": "Connection failed"}, 500