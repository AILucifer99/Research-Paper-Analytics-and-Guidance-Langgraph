
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.models import ResearchState
from src.config import config
from src.utils import ResultsManager

logger = logging.getLogger(__name__)

async def deep_analysis(state: ResearchState) -> ResearchState:
    """Step 6c: Perform quantitative analysis and generate charts"""
    logger.info("Starting deep analysis...")
    
    # We need to find where to save the charts. 
    # Since deep_analysis runs before write_output, we don't have the final run_dir yet.
    # However, we can use a temporary or predictable path, OR we can make write_output rely on this.
    # A better approach for this flow is to let write_output handle directory creation, 
    # but since this is a separate node in the graph, we might want to store the charts in memory 
    # or save them to a temp local dir that write_output can pick up.
    # Alternatively, we can assume write_output will move them or we can just save them to a "latest" folder.
    
    # For simplicity, we will assume we can write to a 'temp_charts' folder and then output_writer will pick them up
    # OR we can modify the state to hold the chart paths/data.
    
    # Let's create a DataFrame from the papers
    all_papers_data = []
    for topic, papers in state.analyzed_papers.items():
        for paper in papers:
            all_papers_data.append({
                'topic': topic,
                'title': paper.title,
                'year': int(paper.published[:4]) if paper.published else None,
                'companies': paper.companies or [],
                'methods': paper.methods or []
            })
            
    if not all_papers_data:
        logger.warning("No data for deep analysis")
        return state

    df = pd.DataFrame(all_papers_data)
    
    # Generate charts
    charts_dir = Path(config.OUTPUT_DIR) / "temp_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Papers per Topic
    plt.figure(figsize=(10, 6))
    df['topic'].value_counts().plot(kind='bar')
    plt.title('Number of Papers per Topic')
    plt.tight_layout()
    plt.savefig(charts_dir / "papers_per_topic.png")
    plt.close()
    
    # 2. Publication Trends (if years available)
    if 'year' in df.columns and df['year'].notna().any():
        plt.figure(figsize=(10, 6))
        df.groupby('year').size().plot(kind='line', marker='o')
        plt.title('Publication Trend')
        plt.tight_layout()
        plt.savefig(charts_dir / "publication_trend.png")
        plt.close()
        
    state.logs.append(f"Generated deep analysis charts in {charts_dir}")
    
    # Ideally, we should add a field to ResearchState to hold chart paths, 
    # but for now we'll just log it. The output_writer can arguably move these if needed,
    # or we can leave them there. 
    # For a robust solution, let's update the output_writer to check for this folder.
    
    return state
