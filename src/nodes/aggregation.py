
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models import ResearchState, TopicStats, CompanyComparison
from src.config import config

logger = logging.getLogger(__name__)

async def attribute_companies(state: ResearchState) -> ResearchState:
    """Step 4: Map author affiliations to companies"""
    logger.info("Starting company attribution...")
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    company_mapping_prompt = ChatPromptTemplate.from_template("""
    Given these author names and paper context, identify the likely company/institution affiliations.
    Consider major tech companies (Google, Microsoft, OpenAI, Meta, etc.), universities, and research labs.
    
    Authors: {authors}
    Paper Title: {title}
    
    Return a JSON list of likely company/institution names. Use canonical names 
    (e.g., "Google" not "Google Inc.", "MIT" not "Massachusetts Institute of Technology").
    If uncertain, return empty list.
    """)
    
    parser = JsonOutputParser()
    chain = company_mapping_prompt | llm | parser
    
    for topic, papers in state.analyzed_papers.items():
        for paper in papers:
            try:
                companies = await chain.ainvoke({
                    "authors": paper.authors,
                    "title": paper.title
                })
                paper.companies = companies if isinstance(companies, list) else []
            except Exception as e:
                logger.error(f"Error attributing companies for '{paper.title}': {e}")
                paper.companies = []
        
        state.logs.append(f"Completed company attribution for '{topic}'")
        
    return state

async def aggregate_topics(state: ResearchState) -> ResearchState:
    """Step 5: Aggregate statistics per topic"""
    logger.info("Starting topic aggregation...")
    
    for topic, papers in state.analyzed_papers.items():
        if not papers:
            continue
            
        # Calculate statistics
        paper_count = len(papers)
        
        # Methods distribution
        methods_dist = {}
        for paper in papers:
            for method in paper.methods or []:
                methods_dist[method] = methods_dist.get(method, 0) + 1
        
        # Company distribution
        company_dist = {}
        for paper in papers:
            for company in paper.companies or []:
                company_dist[company] = company_dist.get(company, 0) + 1
        
        top_companies = sorted(company_dist.keys(), 
                             key=lambda x: company_dist[x], reverse=True)[:5]
        
        # Simplified scores (in real implementation, these would be more sophisticated)
        reproducibility_score = min(1.0, sum(1 for p in papers if p.datasets) / max(1, paper_count))
        open_source_ratio = 0.3  # Placeholder - would need to analyze paper content
        avg_citation_potential = 0.7  # Placeholder - would use actual metrics
        
        topic_stat = TopicStats(
            topic=topic,
            paper_count=paper_count,
            methods_distribution=methods_dist,
            reproducibility_score=reproducibility_score,
            open_source_ratio=open_source_ratio,
            top_companies=top_companies,
            avg_citation_potential=avg_citation_potential
        )
        
        state.topic_stats.append(topic_stat)
        state.logs.append(f"Aggregated statistics for '{topic}': {paper_count} papers")
        
    return state

async def compare_companies(state: ResearchState) -> ResearchState:
    """Step 6: Compare companies across topics"""
    logger.info("Starting company comparison...")
    
    llm = ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2000
    )
    
    # Collect all companies and their involvement
    company_data = {}
    
    for topic_stat in state.topic_stats:
        for company in topic_stat.top_companies:
            if company not in company_data:
                company_data[company] = {
                    'topics': set(),
                    'papers': 0,
                    'methods': set()
                }
            
            company_data[company]['topics'].add(topic_stat.topic)
            
            # Count papers for this company in this topic
            topic_papers = state.analyzed_papers.get(topic_stat.topic, [])
            for paper in topic_papers:
                if company in (paper.companies or []):
                    company_data[company]['papers'] += 1
                    company_data[company]['methods'].update(paper.methods or [])
    
    # Generate comparisons using LLM
    comparison_prompt = ChatPromptTemplate.from_template("""
    Analyze this company's research profile and provide insights:
    
    Company: {company}
    Topics Covered: {topics}
    Total Papers: {paper_count}
    Methods Used: {methods}
    
    Provide analysis in JSON format:
    {{
        "strengths": ["strength 1", "strength 2"],
        "weaknesses": ["weakness 1", "weakness 2"],
        "innovation_score": 0.8,
        "collaboration_score": 0.6
    }}
    
    Scores should be between 0.0 and 1.0.
    """)
    
    parser = JsonOutputParser()
    chain = comparison_prompt | llm | parser
    
    comparisons = []
    
    for company, data in company_data.items():
        try:
            analysis = await chain.ainvoke({
                "company": company,
                "topics": list(data['topics']),
                "paper_count": data['papers'],
                "methods": list(data['methods'])[:10]  # Limit methods for context
            })
            
            comparison = CompanyComparison(
                company=company,
                topics_covered=list(data['topics']),
                paper_count=data['papers'],
                strengths=analysis.get('strengths', []),
                weaknesses=analysis.get('weaknesses', []),
                innovation_score=analysis.get('innovation_score', 0.5),
                collaboration_score=analysis.get('collaboration_score', 0.5),
                ranking=0  # Will be set after sorting
            )
            
            comparisons.append(comparison)
            
        except Exception as e:
            logger.error(f"Error analyzing company '{company}': {e}")
    
    # Rank companies by combined score
    comparisons.sort(key=lambda x: (x.innovation_score + x.collaboration_score), reverse=True)
    for i, comp in enumerate(comparisons):
        comp.ranking = i + 1
    
    state.company_comparisons = comparisons
    state.logs.append(f"Completed comparison for {len(comparisons)} companies")
    
    return state
