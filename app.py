from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from datetime import datetime
from pydantic import BaseModel# FASTAPI standard for data validation
import asyncio

from Kbot import ResearchBot, GeminiProvider, DuckDuckGoSearchProvider

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/ask")
async def ask(question: str):
    llm = GeminiProvider()
    search = DuckDuckGoSearchProvider()
    bot = ResearchBot(llm, search)
    
    # This returns a dictionary: {'answer': '...', 'sources': [...]}
    result = await bot.research(question)
    
    # FIX: Return the result directly. FastAPI converts dict to JSON automatically.
    return result

# 1. Define the Data Model
class FeedbackModel(BaseModel):
    rating: int
    comment: str
    
# 2. The Feedback Endpoint
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackModel):
    file_path = "feedback.json"
    
    # Prepare the new entry with a timestamp
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rating": feedback.rating,
        "comment": feedback.comment,
        "status": "pending" # You can change this to "done" later manually
    }
    
    # Load existing data or create empty list
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
        
    # Append and Save
    data.append(new_entry)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4) # indent=4 makes it readable like a list
        
    return {"message": "Feedback received!"}