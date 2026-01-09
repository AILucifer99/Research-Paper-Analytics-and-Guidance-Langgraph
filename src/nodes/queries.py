
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models import ResearchState
from src.config import config

logger = logging.getLogger(__name__)

async def expand_queries(state: ResearchState) -> ResearchState:
    """Step 1: Expand research topics into search queries"""
    logger.info("Starting query expansion...")
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    expansion_prompt = ChatPromptTemplate.from_template("""
    You are a research expert. Given a research topic, expand it into 3-5 specific search queries 
    that would help find relevant academic papers on arXiv.
    
    Topic: {topic}
    
    Create queries that cover:
    1. Core concepts and methods
    2. Applications and use cases
    3. Recent developments and innovations
    4. Comparative studies
    
    Return as a JSON list of strings.
    """)
    
    parser = JsonOutputParser()
    chain = expansion_prompt | llm | parser
    
    for topic in state.topics:
        try:
            queries = await chain.ainvoke({"topic": topic})
            state.expanded_queries[topic] = queries
            state.logs.append(f"Expanded '{topic}' into {len(queries)} queries")
        except Exception as e:
            logger.error(f"Error expanding topic {topic}: {e}")
            state.expanded_queries[topic] = [topic]  # Fallback
            
    return state
