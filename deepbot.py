"""
ResearchBot - A CLI research assistant using Google APIs
Built with SOLID principles and free Google tools
"""

import os
import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import click
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Google APIs and web scraping
try:
    import google.generativeai as genai
    from googlesearch import search as google_search
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse
except ImportError:
    print("Please install required packages: pip install google-generativeai googlesearch-python requests beautifulsoup4")

load_dotenv()

@dataclass
class SearchResult:
    """Data class for search results - Single Responsibility"""
    title: str
    snippet: str
    url: str
    content: str = ""
    credibility_score: float = 0.0

@dataclass
class SynthesisResult:
    """Data class for synthesis results - Single Responsibility"""
    answer: str
    sources: List[Dict[str, str]]

# SOLID: Interface Segregation Principle
class LLMProvider(ABC):
    @abstractmethod
    async def generate_content(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        pass

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        pass

class ContentExtractor(ABC):
    @abstractmethod
    async def extract_content(self, url: str) -> str:
        pass

# SOLID: Open/Closed Principle - Can extend with new providers
class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        
        # Use the correct model name
        self.model_name = 'gemini-1.5-flash'  # Updated model name
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Using model: {self.model_name}")
        except Exception as e:
            print(f"❌ Model {self.model_name} failed: {e}")
            # Try alternative model names
            try:
                self.model_name = 'models/gemini-pro'
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ Using model: {self.model_name}")
            except:
                raise ValueError("No working Gemini model found")
    
    async def generate_content(self, prompt: str) -> str:
        try:
            # Add safety settings to avoid blocking
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                )
            )
            
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise Exception(f"Content blocked: {response.prompt_feedback.block_reason}")
                
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        json_prompt = f"""
        {prompt}
        
        Respond with ONLY valid JSON in this exact format:
        {{"key": "value"}}
        No other text or explanation.
        """
        response = await self.generate_content(json_prompt)
        
        # Extract JSON from response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            # If JSON parsing fails, try to clean the response
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Last resort - return a simple structure
                return {"answer": response, "sources": []}
        
        raise ValueError("Failed to parse JSON response")

class WebContentExtractor(ContentExtractor):
    """Extracts content with better camouflage to avoid blocking"""
    
    def __init__(self):
        self.session = requests.Session()
        # 🛡️ CAMOUFLAGE: Look like a real Chrome browser on Windows
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
    
    async def extract_content(self, url: str) -> str:
        try:
            print(f"   🌐 Visiting: {url}")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.session.get(url, timeout=15))
            
            if response.status_code == 403:
                return "BLOCKED" # Explicitly mark as blocked
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Clean up the HTML
            for element in soup(["script", "style", "nav", "header", "footer", "iframe", "ads"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            # If text is too short, it's likely a bot protection page or empty
            if len(text) < 300: 
                return "EMPTY"
                
            return text[:4000] # Get MORE data (4000 chars)
            
        except Exception as e:
            return f"ERROR: {str(e)}"

class GoogleSearchProvider(SearchProvider):
    """Google Search provider with web scraping"""
    
    def __init__(self):
        self.content_extractor = WebContentExtractor()
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            results = []
            print(f"🔍 Searching: {query}")
            
            # Use the googlesearch library
            search_results = list(google_search(
                query, 
                num_results=max_results,
                lang="en",
                advanced=True  # Get advanced results with titles and descriptions
            ))
            
            # Extract content for each result
            for i, result in enumerate(search_results):
                try:
                    credibility_score = self._calculate_credibility(result.url)
                    
                    # Extract content from the page
                    content = await self.content_extractor.extract_content(result.url)
                    
                    # Use the actual title from search results if available
                    title = getattr(result, 'title', f"Result from {self._get_domain(result.url)}")
                    
                    results.append(SearchResult(
                        title=title,
                        snippet=getattr(result, 'description', 'No description available'),
                        url=result.url,
                        content=content,
                        credibility_score=credibility_score
                    ))
                    
                    # Small delay to be respectful to servers
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Failed to process {result.url}: {e}")
                    continue
            
            filtered_results = self._filter_credible_results(results)
            print(f"📊 Found {len(filtered_results)} credible results for '{query}'")
            return filtered_results
            
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")
            return []
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        domain = urlparse(url).netloc
        return domain.replace('www.', '')
    
    def _calculate_credibility(self, url: str) -> float:
        """Calculate credibility score based on domain"""
        credible_domains = [
            '.edu', '.gov', '.org', 
            'wikipedia.org', 'bbc.com', 'reuters.com', 
            'ap.org', 'nytimes.com', 'theguardian.com',
            'nature.com', 'science.org', 'arxiv.org'
        ]
        suspicious_domains = ['.blogspot.', '.wordpress.', 'tiktok.com']
        
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in suspicious_domains):
            return 0.3
            
        if any(domain in url_lower for domain in credible_domains):
            return 0.8
            
        return 0.5
    
    def _filter_credible_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on credibility score and content quality"""
        filtered = [r for r in results if r.credibility_score > 0.4 and len(r.content) > 100]
        return filtered
    



class DuckDuckGoSearchProvider(SearchProvider):
    """Smart provider that keeps trying until it finds good data"""
    
    def __init__(self):
        self.content_extractor = WebContentExtractor()
        self.ddgs = DDGS()
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            results = []
            print(f"🔍 Searching via DDG: {query}")
            
            # 1. Fetch MORE links than we need (Pool of candidates)
            # We ask for 10, hoping to find 'max_results' (e.g., 3) good ones
            raw_results = list(self.ddgs.text(query, max_results=10, backend="html"))
            
            success_count = 0
            
            for result in raw_results:
                # Stop if we have enough good sources
                if success_count >= max_results:
                    break
                    
                url = result['href']
                
                # Skip PDF files (hard to scrape text)
                if url.endswith('.pdf'):
                    continue
                    
                try:
                    # 2. Calculate credibility BEFORE scraping
                    credibility_score = self._calculate_credibility(url)
                    
                    # 3. Try to Scrape
                    content = await self.content_extractor.extract_content(url)
                    
                    # 4. Analyze the result
                    if content in ["BLOCKED", "EMPTY"] or content.startswith("ERROR"):
                        print(f"   🚫 Blocked/Empty: {self._get_domain(url)} (Trying next...)")
                        continue # <--- THIS IS THE "TRY NEXT" LOGIC
                    
                    # If we got here, we have GOOD data!
                    print(f"   ✅ Scraped {len(content)} chars from {self._get_domain(url)}")
                    
                    results.append(SearchResult(
                        title=result['title'],
                        snippet=result['body'],
                        url=url,
                        content=content,
                        credibility_score=credibility_score
                    ))
                    
                    success_count += 1
                    await asyncio.sleep(1.5) # Respectful delay
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
            
            print(f"📊 Final Report: gathered {len(results)} good sources from {len(raw_results)} candidates.")
            return results
            
        except Exception as e:
            print(f"❌ Critical Search Error: {e}")
            return []

    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc.replace('www.', '')

    def _calculate_credibility(self, url: str) -> float:
        # Boost known good documentation/news sites
        high_quality = ['.edu', '.gov', 'wikipedia.org', 'reuters.com', 'nature.com', 
                       'stackoverflow.com', 'github.com', 'medium.com', 'towardsdatascience.com']
        low_quality = ['.blogspot.', 'tiktok.com', 'pinterest.com']
        
        if any(d in url.lower() for d in low_quality): return 0.3
        if any(d in url.lower() for d in high_quality): return 0.9
        return 0.6

    # def _filter_credible_results(self, results: List[SearchResult]) -> List[SearchResult]:
    #     return [r for r in results if r.credibility_score > 0.4 and len(r.content) > 100]

# SOLID: Dependency Inversion Principle
class ResearchBot:
    """Main ResearchBot class following Single Responsibility Principle"""
    
    def __init__(self, llm_provider: LLMProvider, search_provider: SearchProvider):
        self.llm = llm_provider
        self.search = search_provider
        self.search_history = []
    
    async def decompose_query(self, question: str) -> Dict[str, Any]:
        """Break down question into sub-queries using LLM"""
        # Simple decomposition without LLM for reliability
        words = question.lower().split()
        key_terms = [word for word in words if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'which']]
        
        if len(key_terms) > 1:
            sub_queries = [
                question,
                f"{' '.join(key_terms)} impact effects",
                f"{' '.join(key_terms)} research study",
                f"{' '.join(key_terms)} recent developments"
            ]
        else:
            sub_queries = [question]
        
        return {
            "sub_queries": sub_queries[:3],  # Max 3 sub-queries
            "reasoning": "Simple keyword-based decomposition for reliability"
        }
    
    async def perform_searches(self, queries: List[str]) -> List[SearchResult]:
        """Perform parallel searches for all queries"""
        tasks = [self.search.search(query, max_results=2) for query in queries]  # Reduced to avoid rate limits
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results = []
        for i, results in enumerate(results_lists):
            if isinstance(results, Exception):
                print(f"❌ Search task {i} failed: {results}")
            elif isinstance(results, list):
                all_results.extend(results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def synthesize_results(self, question: str, results: List[SearchResult]) -> SynthesisResult:
        """Synthesize search results into coherent answer"""
        if not results:
            return SynthesisResult(
                answer="I couldn't find sufficient information to answer your question. Please try rephrasing or using more specific terms.",
                sources=[]
            )
        
        # Prepare context from search results
        context = "\n\n".join([
            f"Source {i+1} (URL: {result.url}):\nContent: {result.content}\nSnippet: {result.snippet}"
            for i, result in enumerate(results[:6])  # Limit context length
        ])
        
        prompt = f"""
        Based on the following search results, answer this question: {question}
        
        Search Results:
        {context}
        
        Guidelines:
        - Be factual, neutral, and concise (200-400 words)
        - Acknowledge limitations if information is insufficient
        - Present balanced view if there are conflicts
        - Use Markdown for readability
        - Focus on the most relevant information
        
        Provide a well-structured answer:
        """
        
        try:
            answer = await self.llm.generate_content(prompt)
            
            # Create sources list
            sources = []
            for i, result in enumerate(results):
                sources.append({
                    "id": i + 1,
                    "url": result.url,
                    "description": f"{result.title} - {self.search._get_domain(result.url)}"
                })
            
            return SynthesisResult(answer=answer, sources=sources[:8])  # Limit sources
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            # Fallback: create basic answer from scraped content
            fallback_answer = self._create_fallback_answer(question, results)
            return SynthesisResult(answer=fallback_answer, sources=[])
    
    def _create_fallback_answer(self, question: str, results: List[SearchResult]) -> str:
        """Create a basic answer when LLM synthesis fails"""
        if not results:
            return "I couldn't find sufficient information to answer your question."
        
        key_points = []
        for i, result in enumerate(results[:3]):
            # Extract first 200 chars as key point
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            key_points.append(f"- From {result.title}: {content_preview}")
        
        return f"""**Answer to: {question}**

Based on my research, here are the key findings:

{"\n".join(key_points)}

*Note: This is a summary based on available sources. For more detailed information, please check the original sources.*"""

    async def research(self, question: str) -> Dict[str, Any]:
        """Main research workflow"""
        print(f"🔍 Researching: {question}")
        
        # Step 1: Query decomposition
        decomposition = await self.decompose_query(question)
        print(f"📋 Sub-queries: {decomposition['sub_queries']}")
        
        # Step 2: Perform searches
        search_results = await self.perform_searches(decomposition['sub_queries'])
        print(f"📚 Found {len(search_results)} total sources with content")
        
        # Step 3: Synthesize results
        synthesis = await self.synthesize_results(question, search_results)
        
        return {
            "question": question,
            "sub_queries": decomposition['sub_queries'],
            "answer": synthesis.answer,
            "sources": synthesis.sources,
            "search_results_count": len(search_results)
        }

# CLI Interface using Click
@click.group()
def cli():
    """ResearchBot - AI-powered research assistant using Google APIs"""
    pass

@cli.command()
@click.argument('question')
@click.option('--api-key', help='Google API Key')
@click.option('--provider', type=click.Choice(['google', 'ddg']), default='ddg', help='Search provider to use') # <--- NEW LINE
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def ask(question, api_key, provider, verbose): # <--- Add 'provider' argument
    """Ask a research question"""
    # Pass provider to main_async
    asyncio.run(main_async(question, api_key, provider, verbose))

@cli.command()
def interactive():
    """Start interactive research session"""
    while True:
        try:
            question = input("\n🎯 Research question (or 'quit' to exit): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                asyncio.run(main_async(question, verbose=True))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def main_async(question: str, api_key: str = None,provider: str = 'ddg', verbose: bool = False):
    """Main async function"""
    try:
        # Initialize providers
        llm_provider = GeminiProvider(api_key)
        
        # Initialize Search Strategy based on user choice
        if provider == 'google':
            print("🕵️  Using Google Search Provider")
            search_provider = GoogleSearchProvider()
        else:
            print("🦆 Using DuckDuckGo Provider")
            search_provider = DuckDuckGoSearchProvider()
        
        # Create research bot
        bot = ResearchBot(llm_provider, search_provider)
        
        # Perform research
        result = await bot.research(question)
        
        # Display results
        print("\n" + "="*60)
        print("🤖 RESEARCH RESULTS")
        print("="*60)
        print(f"Question: {result['question']}\n")
        
        if verbose:
            print(f"Sub-queries: {result['sub_queries']}")
            print(f"Sources analyzed: {result['search_results_count']}\n")
        
        print("Answer:")
        print(result['answer'])
        
        if result['sources']:
            print("\n📚 Sources:")
            for source in result['sources']:
                print(f"[{source['id']}] {source['description']}")
                print(f"    {source['url']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Google API key and internet connection.")

# Simple unit test
def test_research_bot():
    """Basic test function"""
    print("Testing ResearchBot initialization...")
    
    # Test with environment variable
    if os.getenv("GOOGLE_API_KEY"):
        try:
            llm = GeminiProvider()
            search = GoogleSearchProvider()
            bot = ResearchBot(llm, search)
            print("✅ ResearchBot initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    else:
        print("⚠️  GOOGLE_API_KEY not set - skipping full test")
        return False

if __name__ == "__main__":
    # Run basic test
    test_success = test_research_bot()
    print("\n")
    
    if test_success:
        # Start CLI
        cli()
    else:
        print("❌ ResearchBot failed to initialize. Please check your setup.")
        print("\nSetup instructions:")
        print("1. Get a free Google API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GOOGLE_API_KEY='your_key'")
        print("3. Install dependencies: pip install google-generativeai googlesearch-python requests beautifulsoup4 click python-dotenv")

"""
Key Improvements:
1. ✅ Web scraping with BeautifulSoup to extract actual content
2. ✅ Better model selection (gemini-1.0-pro)
3. ✅ Content extraction from search results
4. ✅ Fallback mechanisms when scraping fails
5. ✅ Rate limiting with delays between requests
6. ✅ Better error handling for web scraping
7. ✅ Content quality filtering
8. ✅ Fallback answer generation without LLM

Now the system will:
- Perform actual web searches
- Scrape content from resulting pages
- Extract meaningful text using BeautifulSoup
- Use the actual content for synthesis
- Handle errors gracefully
- Work even if some sites block scraping
"""
