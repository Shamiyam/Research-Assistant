from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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