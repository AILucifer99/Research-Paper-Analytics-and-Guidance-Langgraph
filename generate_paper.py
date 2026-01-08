from data_fetcher.research_paper_pdf_fetcher import ResearchPaperFetcher
from agents.researcher import MultiAgentResearchWriter
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

logger.info("Initializing Research Paper Fetcher...")

research_topic = "Retrieval Augmented Generation, Large Language Models, Video Data"
generated_paper_title = "Video-RAG: Retrieval Augmented Generation with LLMs on Video Content"

paper_save_dir = "research_papers_v2"

fetcher = ResearchPaperFetcher(save_dir=paper_save_dir)

logger.info(f"Fetching papers for topic: {research_topic}")

# Combine multiple filters
papers = fetcher.fetch_and_save(
    topic=research_topic,
    category="cs.AI",
    start_date="2024-06-01",
    max_results=20
)

logger.info(f"Fetched {len(papers)} papers. Printing summary...")
fetcher.print_summary(papers)

logger.info("Initializing MultiAgentResearchWriter...")
# Initialize system
writer = MultiAgentResearchWriter(
    api_key=os.getenv("OPENAI_API_KEY"),
    papers_directory="research_papers_v1",
    model="gpt-4.1-nano",
)

# Generate paper with peer review
logger.info("Starting research paper generation...")
paper = writer.write_research_paper(
    topic=research_topic,
    output_file="my_paper.md",
    review_iterations=1  # Number of improvement cycles
)

logger.info("Research paper generation completed successfully.")