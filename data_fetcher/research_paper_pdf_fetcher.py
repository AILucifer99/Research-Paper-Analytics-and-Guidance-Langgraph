import requests
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

class ResearchPaperFetcher:
    """Fetches top research papers from arXiv and saves PDFs."""
    
    # Popular arXiv categories
    CATEGORIES = {
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cs.CL': 'Computation and Language',
        'cs.CV': 'Computer Vision',
        'cs.NE': 'Neural and Evolutionary Computing',
        'stat.ML': 'Machine Learning (Statistics)',
        'quant-ph': 'Quantum Physics',
        'physics.comp-ph': 'Computational Physics',
        'math.OC': 'Optimization and Control',
        'econ.EM': 'Econometrics',
    }
    
    def __init__(self, save_dir: str = "research_papers"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def search_papers(self, 
                     topic: str, 
                     max_results: int = 10,
                     category: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     min_citations: Optional[int] = None,
                     author: Optional[str] = None) -> List[Dict]:
        """
        Search for papers on arXiv by topic with filters.
        
        Args:
            topic: Search query (e.g., "machine learning", "quantum computing")
            max_results: Number of papers to fetch (default: 10)
            category: arXiv category (e.g., 'cs.AI', 'cs.LG'). Use CATEGORIES dict for options
            start_date: Filter papers after this date (format: 'YYYY-MM-DD')
            end_date: Filter papers before this date (format: 'YYYY-MM-DD')
            min_citations: Minimum citations (Note: arXiv API doesn't support this directly)
            author: Filter by author name
        
        Returns:
            List of dictionaries containing paper metadata
        """
        # Build search query
        query_parts = []
        
        if topic:
            query_parts.append(f'all:{topic}')
        
        if category:
            query_parts.append(f'cat:{category}')
        
        if author:
            query_parts.append(f'au:{author}')
        
        search_query = ' AND '.join(query_parts) if query_parts else 'all:*'
        
        # Construct query parameters
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results * 2,  # Fetch more to account for filtering
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url}?{query_string}"
        
        print(f"Searching for papers...")
        print(f"  Topic: {topic}")
        if category:
            print(f"  Category: {category} ({self.CATEGORIES.get(category, 'Unknown')})")
        if author:
            print(f"  Author: {author}")
        if start_date or end_date:
            print(f"  Date range: {start_date or 'any'} to {end_date or 'any'}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', namespace):
            published_date = entry.find('atom:published', namespace).text[:10]
            
            # Apply date filters
            if start_date and published_date < start_date:
                continue
            if end_date and published_date > end_date:
                continue
            
            paper = {
                'title': entry.find('atom:title', namespace).text.strip().replace('\n', ' '),
                'authors': [author.find('atom:name', namespace).text 
                           for author in entry.findall('atom:author', namespace)],
                'summary': entry.find('atom:summary', namespace).text.strip(),
                'published': published_date,
                'pdf_url': None,
                'arxiv_id': None,
                'categories': []
            }
            
            # Get categories
            for cat in entry.findall('atom:category', namespace):
                paper['categories'].append(cat.get('term'))
            
            # Get PDF URL
            for link in entry.findall('atom:link', namespace):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
            
            # Extract arXiv ID from entry id
            entry_id = entry.find('atom:id', namespace).text
            paper['arxiv_id'] = entry_id.split('/abs/')[-1]
            
            papers.append(paper)
            
            # Stop if we have enough papers
            if len(papers) >= max_results:
                break
        
        print(f"Found {len(papers)} papers matching criteria")
        return papers
    
    def download_pdf(self, paper: Dict) -> str:
        """
        Download PDF for a single paper.
        
        Args:
            paper: Dictionary containing paper metadata
        
        Returns:
            Path to saved PDF file
        """
        if not paper['pdf_url']:
            print(f"No PDF URL for: {paper['title'][:50]}...")
            return None
        
        # Create safe filename
        safe_title = "".join(c for c in paper['title'][:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{paper['arxiv_id']}_{safe_title}.pdf"
        filepath = self.save_dir / filename
        
        print(f"Downloading: {paper['title'][:60]}...")
        
        try:
            response = requests.get(paper['pdf_url'], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Saved to: {filepath}")
            return str(filepath)
        
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            return None
    
    def fetch_and_save(self, 
                      topic: str, 
                      max_results: int = 10,
                      category: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      author: Optional[str] = None) -> List[Dict]:
        """
        Search for papers and download their PDFs with filtering options.
        
        Args:
            topic: Search query
            max_results: Number of papers to fetch
            category: arXiv category filter (e.g., 'cs.AI', 'cs.LG')
            start_date: Filter papers after this date (format: 'YYYY-MM-DD')
            end_date: Filter papers before this date (format: 'YYYY-MM-DD')
            author: Filter by author name
        
        Returns:
            List of papers with download status
        """
        papers = self.search_papers(
            topic=topic,
            max_results=max_results,
            category=category,
            start_date=start_date,
            end_date=end_date,
            author=author
        )
        
        print(f"\nDownloading {len(papers)} PDFs...\n")
        
        for i, paper in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] ", end="")
            filepath = self.download_pdf(paper)
            paper['local_path'] = filepath
            
            # Be nice to the server
            if i < len(papers):
                time.sleep(1)
        
        print(f"\n✓ Complete! Papers saved to: {self.save_dir.absolute()}")
        return papers
    
    def print_summary(self, papers: List[Dict]):
        """Print a summary of fetched papers."""
        print("\n" + "="*80)
        print("PAPER SUMMARY")
        print("="*80)
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}" + 
                  (f" et al." if len(paper['authors']) > 3 else ""))
            print(f"   Published: {paper['published']}")
            print(f"   Categories: {', '.join(paper.get('categories', [])[:3])}")
            print(f"   arXiv ID: {paper['arxiv_id']}")
            print(f"   Status: {'✓ Downloaded' if paper.get('local_path') else '✗ Failed'}")
    
    @staticmethod
    def list_categories():
        """Print available arXiv categories."""
        print("\nAvailable Categories:")
        print("-" * 50)
        for code, name in ResearchPaperFetcher.CATEGORIES.items():
            print(f"  {code:15} - {name}")


# Example usage
if __name__ == "__main__":
    # Create fetcher instance
    fetcher = ResearchPaperFetcher(save_dir="research_papers")
    
    # Show available categories
    ResearchPaperFetcher.list_categories()
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic search")
    print("="*80)
    # Basic search
    papers = fetcher.fetch_and_save("large language models", max_results=5)
    fetcher.print_summary(papers)
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Search with category filter")
    print("="*80)
    # Search with category filter
    papers = fetcher.fetch_and_save(
        topic="transformer",
        max_results=5,
        category="cs.LG"  # Machine Learning
    )
    fetcher.print_summary(papers)
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Search with date range")
    print("="*80)
    # Search with date range
    papers = fetcher.fetch_and_save(
        topic="neural networks",
        max_results=5,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    fetcher.print_summary(papers)
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Search by author")
    print("="*80)
    # Search by specific author
    papers = fetcher.fetch_and_save(
        topic="",
        max_results=5,
        author="Yoshua Bengio"
    )
    fetcher.print_summary(papers)
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Combined filters")
    print("="*80)
    # Combined filters
    papers = fetcher.fetch_and_save(
        topic="reinforcement learning",
        max_results=10,
        category="cs.AI",
        start_date="2024-06-01"
    )
    fetcher.print_summary(papers)