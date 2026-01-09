
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models import ResearchState
from src.config import config

logger = logging.getLogger(__name__)

async def analyze_papers(state: ResearchState) -> ResearchState:
    """Step 3: Analyze papers using LLM"""
    logger.info("Starting paper analysis...")
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    Analyze this research paper and extract key information:
    
    Title: {title}
    Abstract: {abstract}
    
    Provide analysis in JSON format:
    {{
        "summary": "Brief 2-sentence summary",
        "contributions": ["key contribution 1", "key contribution 2"],
        "methods": ["method/technique 1", "method/technique 2"],
        "datasets": ["dataset 1", "dataset 2"],
        "limitations": ["limitation 1", "limitation 2"]
    }}
    """)
    
    parser = JsonOutputParser()
    chain = analysis_prompt | llm | parser
    
    for topic, papers in state.papers.items():
        analyzed_papers = []
        
        for paper in papers:
            try:
                analysis = await chain.ainvoke({
                    "title": paper.title,
                    "abstract": paper.abstract
                })
                
                paper.summary = analysis.get("summary", "")
                paper.contributions = analysis.get("contributions", [])
                paper.methods = analysis.get("methods", [])
                paper.datasets = analysis.get("datasets", [])
                paper.limitations = analysis.get("limitations", [])
                
                analyzed_papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error analyzing paper '{paper.title}': {e}")
                analyzed_papers.append(paper)  # Keep original
        
        state.analyzed_papers[topic] = analyzed_papers
        state.logs.append(f"Analyzed {len(analyzed_papers)} papers for '{topic}'")
        
    return state

async def gap_analysis(state: ResearchState) -> ResearchState:
    """Step 6b: Analyze gaps and future trends"""
    logger.info("Starting gap analysis...")
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    # Collect context
    limitations = []
    for papers in state.analyzed_papers.values():
        for paper in papers:
            limitations.extend(paper.limitations or [])
            
    gap_prompt = ChatPromptTemplate.from_template("""
    As a strategic research advisor, analyze these reported limitations and the current state of research.
    
    Reported Limitations: {limitations}
    Topics: {topics}
    
    Identify:
    1. Major research gaps (what is missing?)
    2. Emerging trends based on what is being attempted
    3. Recommendations for future research directions
    
    Return a JSON object with keys: "gaps", "trends", "recommendations" (each a list of strings).
    """)
    
    parser = JsonOutputParser()
    chain = gap_prompt | llm | parser
    
    try:
        # Sample limitations if too many to fit context
        sample_limitations = limitations[:30] if len(limitations) > 30 else limitations
        
        insights = await chain.ainvoke({
            "limitations": sample_limitations,
            "topics": state.topics
        })
        
        state.strategic_insights = insights
        state.logs.append("Completed gap analysis and strategic insights")
        
    except Exception as e:
        logger.error(f"Error during gap analysis: {e}")
        state.strategic_insights = {}
        
    return state
