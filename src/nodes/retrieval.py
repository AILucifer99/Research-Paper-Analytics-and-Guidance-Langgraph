
import logging
import arxiv
from src.models import ResearchState, Paper

logger = logging.getLogger(__name__)

async def retrieve_papers(state: ResearchState) -> ResearchState:
    """Step 2: Retrieve papers from arXiv"""
    logger.info("Starting paper retrieval...")
    
    client = arxiv.Client()
    
    for topic, queries in state.expanded_queries.items():
        papers = []
        
        for query in queries[:3]:  # Limit queries to avoid rate limits
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=10,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for result in client.results(search):
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        affiliations=[],  # arXiv doesn't provide detailed affiliations
                        abstract=result.summary,
                        published=result.published.isoformat(),
                        url=result.entry_id,
                        categories=[cat for cat in result.categories]
                    )
                    papers.append(paper)
                    
            except Exception as e:
                logger.error(f"Error retrieving papers for query '{query}': {e}")
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        for paper in papers:
            if paper.title not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(paper.title)
        
        state.papers[topic] = unique_papers
        state.logs.append(f"Retrieved {len(unique_papers)} unique papers for '{topic}'")
        
    return state
