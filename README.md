# Research-Assistant
Research Assistant That can answer anything...

"""
# ResearchBot

A free, open-source research assistant using Google's Gemini AI and web search.

## Setup

1. Get a free Google API key from: https://makersuite.google.com/app/apikey
2. Install dependencies:
   ```bash
   pip install google-generativeai googlesearch-python click python-dotenv duckduckgo-search trafilatura

3. Set environment variable-
export GOOGLE_API_KEY="your_google_api_key"

4. # Single question
python research_bot.py ask "What is the impact of AI on climate change?"

# Interactive mode
python research_bot.py interactive

# With API key
python research_bot.py ask "Your question" --api-key "your_key"

## Key Improvements Made:

### 1. **Free Google-Based Tools**
- **Google Gemini AI** instead of OpenAI (free tier available)
- **googlesearch-python** for web search (free)


### 2. **SOLID Principles Implementation**
- **S** (Single Responsibility): Each class has one clear purpose
- **O** (Open/Closed): Easy to extend with new providers without modifying existing code
- **L** (Liskov Substitution): All providers are interchangeable
- **I** (Interface Segregation): Separate interfaces for LLM and Search
- **D** (Dependency Inversion): High-level modules depend on abstractions

### 3. **Enhanced Architecture**
- Abstract base classes for easy testing and extension
- Dependency injection for better testability
- Data classes for type safety
- Proper error handling and fallbacks

### 4. **Cost Optimization**
- Uses free Google Gemini tier
- Free web search API
- No external paid dependencies

### 5. **Google Pro Account Benefits**
- Higher rate limits with Google API key
- Better reliability than completely free services
- Access to Google's latest AI models

The code maintains all original functionality while being completely free to use and following software engineering best practices with SOLID principles.