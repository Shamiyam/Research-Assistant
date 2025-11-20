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
import trafilatura
from ddgs import DDGS

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
        
        possible_models = [
                
                'gemini-2.5-flash',          # Latest stable: Fast, intelligent, multimodal (recommended for most)
                'gemini-2.5-flash-lite',     # Ultra-fast, cost-efficient
                'gemini-2.0-flash',          # Second-gen workhorse, 1M tokens
                'gemini-2.0-flash-lite',     # Low-latency variant
                'gemini-2.5-pro',            # Advanced reasoning (if you need deeper synthesis)
                'gemini-2.0-flash-001',      # Tuned stable release (fallback)
                'gemini-1.5-pro-latest'      # Last 1.x resort (if API still supports)
                
            ]
        self.model = None
        for model_name in possible_models:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                print(f"✅ Using model: {self.model_name}")
                break
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue
        if not self.model:
            raise ValueError("No working Gemini model found. Check API key/version.")    
    
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
                    temperature=0.3,
                    top_p=0.95,
                    top_k=50,
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
    """Extracts main article content using Trafilatura (ignores menus/garbage)"""
    
    def __init__(self):
        # We don't need a session for trafilatura's fetch_url, 
        # but we keep requests for backup if needed.
        pass
    
    async def extract_content(self, url: str) -> str:
        try:
            print(f"   🌐 Visiting: {url}")
            
            # Trafilatura download (Run in executor because it is synchronous)
            loop = asyncio.get_event_loop()
            downloaded = await loop.run_in_executor(None, lambda: trafilatura.fetch_url(url))
            
            if not downloaded:
                return "EMPTY" # Site blocked or unreachable
            
            # Extract the MAIN TEXT only (ignores headers, footers, menus)
            # include_comments=False removes user comments which are often garbage
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, favor="fast")        
            if not text:
                return "EMPTY" # Trafilatura couldn't find a main article
                
            # Clean up formatting
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Double Check: If the text is too short, it's probably just a copyright notice
            if len(text) < 200:
                return "EMPTY"
                
            return text[:5000] # Return a nice chunk of valid text
            
        except Exception as e:
            return f"ERROR: {str(e)}"

class GoogleSearchProvider(SearchProvider):
    """Google Search provider with web scraping"""
    
    def __init__(self):
        self.content_extractor = WebContentExtractor()
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            results = []
            print(f"🔍 Searching via Google: {query}")
            
            # Step 1: Dynamic enhancement (build advanced query string for google_search)
            cleaned = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
            words = [w for w in cleaned.split() if len(w) >= 2]
            key_phrases = [query]  # Exact
            if len(words) >= 2:
                potential_phrases = [' '.join(words[i:i+2]) for i in range(0, len(words), 2)]
                key_phrases.extend(potential_phrases[:2])
            
            # Build enhanced query (e.g., "keyphrase1" OR "keyphrase2")
            enhanced_parts = [f'"{phrase}"' for phrase in key_phrases[:3]]
            enhanced_query = ' OR '.join(enhanced_parts)
            
            # Conditional credible sites
            research_words = {'impact', 'effects', 'research', 'study', 'why', 'how', 'analysis'}
            if any(word in query.lower() for word in research_words):
                enhanced_query += ' site:.edu OR site:.gov OR site:.org OR site:un.org OR site:who.int OR site:arxiv.org'
                print(f"📚 Research mode: Added credible site filters")
            
            # Recency (google_search supports 'daterange:' but rough; use num_results filter)
            if 'recent' in query.lower() or '202' in query:
                print("🕐 Recent mode: Limiting to fresh results")
            
            # Use google_search (num_results=10 for filtering)
            search_results = list(google_search(enhanced_query, num_results=10, lang="en", advanced=True))
            
            success_count = 0
            for result in search_results:
                if success_count >= max_results:
                    break
                
                url = result.url  # Note: google_search result has .url, not 'href'
                if url.endswith('.pdf'):
                    continue
                
                title_lower = getattr(result, 'title', '').lower()
                snippet_lower = getattr(result, 'description', '').lower()  # google_search uses 'description'
                
                # Noise filter (adapted for google_search attrs)
                noise_indicators = {
                    'grammar': ['grammar', 'preposition', 'verb', 'syntax'],
                    'ad': ['buy', 'sale', 'sponsor', 'affiliate'],
                    'spam': ['click here', 'free download', 'conspiracy']
                }
                skip = False
                for category, indicators in noise_indicators.items():
                    if any(ind in title_lower or ind in snippet_lower for ind in indicators):
                        if not any(ind in query.lower() for ind in indicators):
                            print(f"🚫 Skipped {category} noise: {getattr(result, 'title', 'Unknown')[:50]}")
                            skip = True
                            break
                if skip:
                    continue
                
                # Extract content
                content = await self.content_extractor.extract_content(url)
                if content in ["EMPTY"] or content.startswith("ERROR"):  # No "BLOCKED" in trafilatura
                    print(f" 🚫 No article found at {self._get_domain(url)} (Trafilatura rejected it)")
                    credibility = self._calculate_credibility(url)
                    if len(results) == 0 and credibility > 0.6:
                        print(f" ⚠️ Fallback: Using snippet for {self._get_domain(url)}")
                        results.append(SearchResult(
                            title=getattr(result, 'title', 'No title'),
                            snippet=getattr(result, 'description', 'No description'),
                            url=url,
                            content=f"Summary: {getattr(result, 'description', '')}",
                            credibility_score=credibility
                        ))
                    continue
                
                print(f" ✅ Scraped {len(content)} chars from {self._get_domain(url)}")
                results.append(SearchResult(
                    title=getattr(result, 'title', 'No title'),
                    snippet=getattr(result, 'description', 'No description'),
                    url=url,
                    content=content,
                    credibility_score=self._calculate_credibility(url)
                ))
                success_count += 1
                await asyncio.sleep(1)  # Rate limit
            
            # Filter credible (your original)
            filtered_results = self._filter_credible_results(results)
            print(f"📊 Final Report: gathered {len(filtered_results)} sources.")
            return filtered_results
            
        except Exception as e:
            print(f"❌ Critical Search Error: {e}")
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
    """Smart provider: Tries to scrape, but falls back to snippets if scraping fails"""
    
    def __init__(self):
        self.content_extractor = WebContentExtractor()
        self.ddgs = DDGS()
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            results = []
            print(f"🔍 Searching via DDG: {query}")
            
            # Fetch candidate links
            raw_results = list(self.ddgs.text(query, max_results=10, backend="html"))
            
            success_count = 0
            
            for result in raw_results:
                if success_count >= max_results:
                    break
                
                url = result['href']
                if url.endswith('.pdf'): continue
                
                # Try to get the FULL MEAL (Article text)
                content = await self.content_extractor.extract_content(url)
                
                # Logic: Did we get a meal, or just an empty plate?
                if content in ["BLOCKED", "EMPTY"] or content.startswith("ERROR"):
                    print(f"   🚫 No article found at {self._get_domain(url)} (Trafilatura rejected it)")
                    
                    # --- THE FALLBACK ---
                    # If we are desperate (have 0 results) and this is a high credibility site,
                    # we will accept the Snippet instead of nothing.
                    credibility = self._calculate_credibility(url)
                    if len(results) == 0 and credibility > 0.6:
                        print(f"   ⚠️ Fallback: Using snippet for {self._get_domain(url)}")
                        results.append(SearchResult(
                            title=result['title'],
                            snippet=result['body'],
                            url=url,
                            content=f"Summary: {result['body']}", # Use snippet as content
                            credibility_score=credibility
                        ))
                        # We don't increment success_count here, we keep looking for better data
                    continue 
                
                # We found a REAL article!
                print(f"   ✅ Scraped {len(content)} chars from {self._get_domain(url)}")
                results.append(SearchResult(
                    title=result['title'],
                    snippet=result['body'],
                    url=url,
                    content=content,
                    credibility_score=self._calculate_credibility(url)
                ))
                success_count += 1
                await asyncio.sleep(1)

            print(f"📊 Final Report: gathered {len(results)} sources.")
            return results
            
        except Exception as e:
            print(f"❌ Critical Search Error: {e}")
            return []

    # (Keep your helper methods: _get_domain, _calculate_credibility, etc.)
    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc.replace('www.', '')

    def _calculate_credibility(self, url: str) -> float:
        high_quality = ['.edu', '.gov', 'wikipedia.org', 'reuters.com', 'nature.com', 'medium.com']
        low_quality = ['.blogspot.', 'tiktok.com']
        if any(d in url.lower() for d in low_quality): return 0.3
        if any(d in url.lower() for d in high_quality): return 0.9
        return 0.6
    # def _filter_credible_results(self, results: List[SearchResult]) -> List[SearchResult]:
    #     return [r for r in results if r.credibility_score > 0.4 and len(r.content) > 100]

# SOLID: Dependency Inversion Principle
# SOLID: Dependency Inversion Principle
class ResearchBot:
    """Main ResearchBot class following Single Responsibility Principle"""
    
    def __init__(self, llm_provider: LLMProvider, search_provider: SearchProvider):
        self.llm = llm_provider
        self.search = search_provider
        self.search_history = []
    
    async def decompose_query(self, question: str) -> Dict[str, Any]:
        """
        Uses LLM to decompose the query from First Principles.
        Analyzes intent -> Identifies Relationships -> Generates Targeted Queries.
        """
        
        # FIRST PRINCIPLES PROMPT
        # We ask the LLM to think before it searches.
        prompt = f"""
        You are an expert Research Planner. 
        User Question: "{question}"

        Task: Break this question down to find the exact answer. 
        
        CRITICAL THINKING PROCESS:
        1. Analyze the user's intent. Are they asking for a specific fact (name, date), a concept, or recent news?
        2. If the user connects two distinct concepts (e.g., "Son" and "Plane"), DO NOT search for them separately. Search for the RELATIONSHIP between them.
        3. Avoid generic keywords that lead to irrelevant viral news (e.g., if asking about family, explicitly avoid "tracker" or "scandal" unless relevant).

        Generate 3 specific search queries in JSON format:
        - Query 1: Direct answer search (The most probable specific search).
        - Query 2: Contextual search (Searching for the story or relationship).
        - Query 3: Alternative phrasing (To catch different sources).

        Output ONLY valid JSON in this format:
        {{
            "reasoning": "Brief explanation of the strategy",
            "sub_queries": ["query string 1", "query string 2", "query string 3"]
        }}
        """
        
        try:
            # Use the LLM to generate the plan directly
            plan = await self.llm.generate_json(prompt)
            
            return {
                "sub_queries": plan.get("sub_queries", [question]),
                "reasoning": plan.get("reasoning", "LLM Generated Plan"),
                "raw_sub_queries": plan.get("sub_queries", [])
            }
            
        except Exception as e:
            print(f"⚠️ Planning failed ({e}), falling back to raw search.")
            # Fallback if Gemini fails to return JSON
            return {
                "sub_queries": [question, f"{question} details", f"{question} explanation"],
                "reasoning": "Fallback: Raw search",
                "raw_sub_queries": []
            }

    async def perform_searches(self, queries: List[str]) -> List[SearchResult]:
        """Perform parallel searches for all queries"""
        # Deduplicate queries to save time/API calls
        unique_queries = list(set(queries))
        
        print(f"🔎 Executing Search Plan: {unique_queries}")
        
        tasks = [self.search.search(query, max_results=2) for query in unique_queries]
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
        
        # ENHANCED SYNTHESIS PROMPT
        prompt = f"""
        You are a professional research assistant.
        
        User Question: {question}
        
        Strict Instructions:
        1. Answer the question DIRECTLY based ONLY on the provided sources.
        2. If the sources discuss "Jet Tracking" but the user asked about a "Son", explicitly state that the search results might be mixed but try to find the name.
        3. Format with Markdown (Bold key terms, use bullet points).
        4. Cite sources using [Source X].
        
        Search Data:
        {context}
        """
        
        try:
            answer = await self.llm.generate_content(prompt)
            
            # Create sources list
            sources = []
            for i, result in enumerate(results):
                try:
                    domain = self.search._get_domain(result.url)
                except AttributeError:
                    domain = urlparse(result.url).netloc.replace('www.', '')
                sources.append({
                    "id": i + 1,
                    "url": result.url,
                    "description": f"{result.title} - {domain}"
                })
            
            return SynthesisResult(answer=answer, sources=sources[:8])
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            fallback_answer = self._create_fallback_answer(question, results)
            return SynthesisResult(answer=fallback_answer, sources=[])
    
    def _create_fallback_answer(self, question: str, results: List[SearchResult]) -> str:
        """Create a basic answer when LLM synthesis fails"""
        if not results:
            return "I couldn't find sufficient information to answer your question."
        
        key_points = []
        for i, result in enumerate(results[:3]):
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            key_points.append(f"- From {result.title}: {content_preview}")
        
        return f"""**Answer to: {question}**

Based on my research, here are the key findings:

{"\n".join(key_points)}

*Note: This is a summary based on available sources. For more detailed information, please check the original sources.*"""

    async def research(self, question: str) -> Dict[str, Any]:
        """Main research workflow"""
        print(f"🔍 Researching: {question}")
        
        # Step 1: Query decomposition (Now AI-Powered)
        decomposition = await self.decompose_query(question)
        print(f"🧠 Planner Reasoning: {decomposition['reasoning']}")
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
    print("Testing ResearchBot initialization...")
    if os.getenv("GOOGLE_API_KEY"):
        try:
            llm = GeminiProvider()
            search = DuckDuckGoSearchProvider()  # <-- Swap to DDG for consistency
            bot = ResearchBot(llm, search)
            print("✅ ResearchBot initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    else:
        print("⚠️ GOOGLE_API_KEY not set - skipping full test")
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
