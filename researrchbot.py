
import os
import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import click
from dotenv import load_dotenv

# Google APIs
try:
    import google.generativeai as genai
    from googlesearch import search as google_search
except ImportError:
    print("Please install required packages: pip install google-generativeai googlesearch-python")

load_dotenv()

@dataclass
class SearchResult:
    """Data class for search results - Single Responsibility"""
    title: str
    snippet: str
    url: str
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

# SOLID: Open/Closed Principle - Can extend with new providers
class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate_content(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
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
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("Failed to parse JSON response")

class GoogleSearchProvider(SearchProvider):
    """Google Custom Search provider using free googlesearch-python"""
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        try:
            results = []
            search_results = google_search(
                query, 
                num_results=max_results,
                lang="en"
            )
            
            for i, url in enumerate(search_results):
                # For free tier, we get URLs only. We'll use URL for credibility scoring
                credibility_score = self._calculate_credibility(url)
                results.append(SearchResult(
                    title=f"Result {i+1}",
                    snippet=f"Content from {url}",
                    url=url,
                    credibility_score=credibility_score
                ))
            
            return self._filter_credible_results(results)
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _calculate_credibility(self, url: str) -> float:
        """Calculate credibility score based on domain"""
        credible_domains = ['.edu', '.gov', '.org', 'wikipedia.org', 'bbc.com', 'reuters.com']
        if any(domain in url.lower() for domain in credible_domains):
            return 0.8
        return 0.5
    
    def _filter_credible_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on credibility score"""
        return [r for r in results if r.credibility_score > 0.6]

# SOLID: Dependency Inversion Principle
class ResearchBot:
    """Main ResearchBot class following Single Responsibility Principle"""
    
    def __init__(self, llm_provider: LLMProvider, search_provider: SearchProvider):
        self.llm = llm_provider
        self.search = search_provider
        self.search_history = []
    
    async def decompose_query(self, question: str) -> Dict[str, Any]:
        """Break down question into sub-queries using LLM"""
        prompt = f"""
        Analyze the following research question and break it down into 2-5 targeted search queries.
        Ensure queries are specific, diverse, and use search operators where helpful.
        
        Question: {question}
        
        Respond with JSON in this exact format:
        {{
            "sub_queries": ["query1", "query2", ...],
            "reasoning": "brief explanation of decomposition"
        }}
        """
        
        return await self.llm.generate_json(prompt)
    
    async def perform_searches(self, queries: List[str]) -> List[SearchResult]:
        """Perform parallel searches for all queries"""
        tasks = [self.search.search(query, max_results=3) for query in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results = []
        for results in results_lists:
            if isinstance(results, list):
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
            f"Source {i+1}:\nURL: {result.url}\nContent: {result.snippet}"
            for i, result in enumerate(results)
        ])
        
        prompt = f"""
        Question: {question}
        
        Search Results:
        {context}
        
        Based on the search results above, synthesize a comprehensive answer following these rules:
        1. Be factual, neutral, and transparent about limitations
        2. Cite sources inline using [1], [2], etc.
        3. If information is conflicting, acknowledge this and present balanced view
        4. If information is insufficient, state this clearly
        5. Keep answer concise (200-400 words)
        6. Use Markdown formatting for readability
        
        Respond with JSON in this exact format:
        {{
            "answer": "Your synthesized answer with citations in markdown format...",
            "sources": [
                {{"id": 1, "url": "https://example.com", "description": "Brief description"}},
                ...
            ]
        }}
        """
        
        return await self.llm.generate_json(prompt)
    
    async def research(self, question: str) -> Dict[str, Any]:
        """Main research workflow"""
        print(f"🔍 Researching: {question}")
        
        # Step 1: Query decomposition
        decomposition = await self.decompose_query(question)
        print(f"📋 Sub-queries: {decomposition['sub_queries']}")
        
        # Step 2: Perform searches
        search_results = await self.perform_searches(decomposition['sub_queries'])
        print(f"📚 Found {len(search_results)} credible sources")
        
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
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def ask(question, api_key, verbose):
    """Ask a research question"""
    asyncio.run(main_async(question, api_key, verbose))

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

async def main_async(question: str, api_key: str = None, verbose: bool = False):
    """Main async function"""
    try:
        # Initialize providers
        llm_provider = GeminiProvider(api_key)
        search_provider = GoogleSearchProvider()
        
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
                print(f"[{source['id']}] {source.get('description', 'No description')}")
                print(f"    {source['url']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Google API key and internet connection.")

# Simple unit test
def test_research_bot():
    """Basic test function"""
    # This would require mocking in a real implementation
    print("Testing ResearchBot initialization...")
    
    # Test with environment variable
    if os.getenv("GOOGLE_API_KEY"):
        try:
            llm = GeminiProvider()
            search = GoogleSearchProvider()
            bot = ResearchBot(llm, search)
            print("✅ ResearchBot initialized successfully")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
    else:
        print("⚠️  GOOGLE_API_KEY not set - skipping full test")

if __name__ == "__main__":
    # Run basic test
    test_research_bot()
    print("\n")
    
    # Start CLI
    cli()

