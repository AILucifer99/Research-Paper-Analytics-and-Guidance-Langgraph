
import logging
from langgraph.graph import StateGraph, END
from src.models import ResearchState

# Import nodes
from src.nodes.queries import expand_queries
from src.nodes.retrieval import retrieve_papers
from src.nodes.analysis import analyze_papers, gap_analysis
from src.nodes.aggregation import attribute_companies, aggregate_topics, compare_companies
from src.nodes.deep_analysis import deep_analysis
from src.nodes.output import write_output

logger = logging.getLogger(__name__)

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow"""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("query_expansion", expand_queries)
    workflow.add_node("paper_retrieval", retrieve_papers)
    workflow.add_node("paper_analysis", analyze_papers)
    workflow.add_node("company_attribution", attribute_companies)
    workflow.add_node("topic_aggregation", aggregate_topics)
    workflow.add_node("company_comparison", compare_companies)
    workflow.add_node("deep_analysis", deep_analysis)
    workflow.add_node("gap_analysis", gap_analysis)
    workflow.add_node("output_writer", write_output)
    
    # Define edges
    workflow.add_edge("query_expansion", "paper_retrieval")
    workflow.add_edge("paper_retrieval", "paper_analysis")
    workflow.add_edge("paper_analysis", "company_attribution")
    workflow.add_edge("company_attribution", "topic_aggregation")
    workflow.add_edge("topic_aggregation", "company_comparison")
    workflow.add_edge("company_comparison", "deep_analysis")
    workflow.add_edge("deep_analysis", "gap_analysis")
    workflow.add_edge("gap_analysis", "output_writer")
    workflow.add_edge("output_writer", END)
    
    # Set entry point
    workflow.set_entry_point("query_expansion")
    
    return workflow.compile()

class ResearchAnalysisSystem:
    def __init__(self, google_api_key: str = None, model_name: str = None):
        # API key and model name are now handled by config.py, 
        # but we keep init args for backward compatibility/overrides if needed
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize the workflow graph
        self.workflow = build_workflow()

    async def analyze_topics(self, topics: list) -> dict:
        """Main entry point for the analysis system"""
        self.logger.info(f"Starting analysis for topics: {topics}")
        
        # Initialize state
        initial_state = ResearchState(topics=topics)
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract output directory from logs if available
        output_dir = "research_output"
        for log in final_state["logs"][::-1]:
            if log.startswith("OUTPUT_DIR:"):
                output_dir = log.split(":", 1)[1]
                break

        return {
            "topics_analyzed": len(final_state["topics"]),
            "papers_found": sum(len(papers) for papers in final_state["analyzed_papers"].values()),
            "companies_identified": len(final_state["company_comparisons"]),
            "output_directory": output_dir,
            "execution_logs": final_state["logs"]
        }
