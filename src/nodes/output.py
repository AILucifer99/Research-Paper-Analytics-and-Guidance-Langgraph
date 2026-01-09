
import logging
import json
from dataclasses import asdict
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models import ResearchState
from src.config import config
from src.utils import ResultsManager
from src.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

async def write_output(state: ResearchState) -> ResearchState:
    """Step 7: Write structured output and logs"""
    logger.info("Writing output files...")
    
    results_manager = ResultsManager(config.OUTPUT_DIR)
    
    # Create run directory
    run_dir = results_manager.create_run_directory()
    
    # Prepare output data
    output_data = {
        "metadata": {
            "timestamp": run_dir.name.split('_', 1)[1],
            "topics_analyzed": len(state.topics),
            "total_papers": sum(len(papers) for papers in state.analyzed_papers.values()),
            "companies_found": len(state.company_comparisons),
            "strategic_insights": state.strategic_insights
        },
        "topics": state.topics,
        "topic_statistics": [asdict(stat) for stat in state.topic_stats],
        "company_comparisons": [asdict(comp) for comp in state.company_comparisons],
        "papers": {
            topic: [asdict(paper) for paper in papers]
            for topic, papers in state.analyzed_papers.items()
        }
    }
    
    # Move temp charts if they exist
    temp_charts_dir = Path(config.OUTPUT_DIR) / "temp_charts"
    if temp_charts_dir.exists():
        final_charts_dir = run_dir / "charts"
        final_charts_dir.mkdir(parents=True, exist_ok=True)
        for chart in temp_charts_dir.glob("*.png"):
            chart.rename(final_charts_dir / chart.name)
        # Clean up temp dir
        try:
            temp_charts_dir.rmdir()
        except:
            pass # Directory might not be empty or locked
            
    # Write JSON output
    output_file = run_dir / "analysis.json"
    results_manager.save_json(output_data, output_file)
    
    # Write execution log
    log_file = run_dir / "execution.log"
    log_content = f"Research Analysis Execution Log\n{'='*50}\n\n" + "\n".join(state.logs)
    results_manager.save_text(log_content, log_file)
    
    # Write summary report
    summary_file = run_dir / "summary_report.md"
    await _generate_summary_report(state, summary_file, results_manager)
    
    # Generate PDF Report
    try:
        pdf_gen = ReportGenerator()
        pdf_file = run_dir / "research_report.pdf"
        
        # Check if we have charts to include
        charts_dir = run_dir / "charts" if (run_dir / "charts").exists() else None
        
        pdf_gen.generate_report(output_data, pdf_file, charts_dir)
        state.logs.append(f"PDF Report generated: {pdf_file}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        state.logs.append(f"PDF generation failed: {e}")
    
    state.logs.append(f"Output written to: {run_dir}")
    logger.info(f"Analysis complete. Results saved to {run_dir}")
    
    # Save output directory in state for return value
    # We use a special marker that can be parsed later if needed, 
    # but ideally we should update the state model to include output_dir
    state.logs.append(f"OUTPUT_DIR:{run_dir}") 

    return state

async def _generate_summary_report(state: ResearchState, file_path: Path, results_manager: ResultsManager):
    """Generate a human-readable summary report"""
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    report_prompt = ChatPromptTemplate.from_template("""
    Generate a comprehensive markdown research report based on this analysis:
    
    Topics Analyzed: {topics}
    Total Papers: {total_papers}
    Top Companies: {top_companies}
    
    Topic Statistics: {topic_stats}
    Company Rankings: {company_rankings}
    
    Create a professional report with:
    1. Executive Summary
    2. Key Findings
    3. Topic Analysis
    4. Company Comparison
    5. Recommendations
    
    Use markdown formatting with proper headers, bullet points, and emphasis.
    """)
    
    chain = report_prompt | llm | StrOutputParser()
    
    try:
        report = await chain.ainvoke({
            "topics": state.topics,
            "total_papers": sum(len(papers) for papers in state.analyzed_papers.values()),
            "top_companies": [comp.company for comp in state.company_comparisons[:5]],
            "topic_stats": [f"{stat.topic}: {stat.paper_count} papers" for stat in state.topic_stats],
            "company_rankings": [f"{comp.ranking}. {comp.company}" for comp in state.company_comparisons[:10]]
        })
        
        results_manager.save_text(report, file_path)
            
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
