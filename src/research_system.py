import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from pathlib import Path

# Core imports
import arxiv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import Graph, StateGraph, END
from dotenv import load_dotenv

from .models import ResearchState, Paper, TopicStats, CompanyComparison
from .utils import ResultsManager
from .report_generator import ReportGenerator

load_dotenv(override=True)

class ResearchAnalysisSystem:
    def __init__(self, google_api_key: str, model_name: str, output_dir: str = "research_output"):
        self.llm_creative = ChatGoogleGenerativeAI(
            api_key=google_api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        self.llm_analytical = ChatGoogleGenerativeAI(
            api_key=google_api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        self.llm_structured = ChatGoogleGenerativeAI(
            api_key=google_api_key,
            model=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        self.results_manager = ResultsManager(output_dir)
        
        # Setup logging - logic will happen when we start a run
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("query_expansion", self._expand_queries)
        workflow.add_node("paper_retrieval", self._retrieve_papers)
        workflow.add_node("paper_analysis", self._analyze_papers)
        workflow.add_node("company_attribution", self._attribute_companies)
        workflow.add_node("topic_aggregation", self._aggregate_topics)
        workflow.add_node("company_comparison", self._compare_companies)
        workflow.add_node("gap_analysis", self._gap_analysis)
        workflow.add_node("output_writer", self._write_output)
        
        # Define edges
        workflow.add_edge("query_expansion", "paper_retrieval")
        workflow.add_edge("paper_retrieval", "paper_analysis")
        workflow.add_edge("paper_analysis", "company_attribution")
        workflow.add_edge("company_attribution", "topic_aggregation")
        workflow.add_edge("topic_aggregation", "company_comparison")
        workflow.add_edge("company_comparison", "gap_analysis")
        workflow.add_edge("gap_analysis", "output_writer")
        workflow.add_edge("output_writer", END)
        
        # Set entry point
        workflow.set_entry_point("query_expansion")
        
        return workflow.compile()

    async def _expand_queries(self, state: ResearchState) -> ResearchState:
        """Step 1: Expand research topics into search queries"""
        self.logger.info("Starting query expansion...")
        
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
        chain = expansion_prompt | self.llm_creative | parser
        
        for topic in state.topics:
            try:
                queries = await chain.ainvoke({"topic": topic})
                state.expanded_queries[topic] = queries
                state.logs.append(f"Expanded '{topic}' into {len(queries)} queries")
            except Exception as e:
                self.logger.error(f"Error expanding topic {topic}: {e}")
                state.expanded_queries[topic] = [topic]  # Fallback
                
        return state

    async def _retrieve_papers(self, state: ResearchState) -> ResearchState:
        """Step 2: Retrieve papers from arXiv"""
        self.logger.info("Starting paper retrieval...")
        
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
                    self.logger.error(f"Error retrieving papers for query '{query}': {e}")
            
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

    async def _analyze_papers(self, state: ResearchState) -> ResearchState:
        """Step 3: Analyze papers using LLM"""
        self.logger.info("Starting paper analysis...")
        
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
        chain = analysis_prompt | self.llm_analytical | parser
        
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
                    self.logger.error(f"Error analyzing paper '{paper.title}': {e}")
                    analyzed_papers.append(paper)  # Keep original
            
            state.analyzed_papers[topic] = analyzed_papers
            state.logs.append(f"Analyzed {len(analyzed_papers)} papers for '{topic}'")
            
        return state

    async def _attribute_companies(self, state: ResearchState) -> ResearchState:
        """Step 4: Map author affiliations to companies"""
        self.logger.info("Starting company attribution...")
        
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
        chain = company_mapping_prompt | self.llm_structured | parser
        
        for topic, papers in state.analyzed_papers.items():
            for paper in papers:
                try:
                    companies = await chain.ainvoke({
                        "authors": paper.authors,
                        "title": paper.title
                    })
                    paper.companies = companies if isinstance(companies, list) else []
                except Exception as e:
                    self.logger.error(f"Error attributing companies for '{paper.title}': {e}")
                    paper.companies = []
            
            state.logs.append(f"Completed company attribution for '{topic}'")
            
        return state

    async def _aggregate_topics(self, state: ResearchState) -> ResearchState:
        """Step 5: Aggregate statistics per topic"""
        self.logger.info("Starting topic aggregation...")
        
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

    async def _compare_companies(self, state: ResearchState) -> ResearchState:
        """Step 6: Compare companies across topics"""
        self.logger.info("Starting company comparison...")
        
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
        chain = comparison_prompt | self.llm_analytical | parser
        
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
                self.logger.error(f"Error analyzing company '{company}': {e}")
        
        # Rank companies by combined score
        comparisons.sort(key=lambda x: (x.innovation_score + x.collaboration_score), reverse=True)
        for i, comp in enumerate(comparisons):
            comp.ranking = i + 1
        
        state.company_comparisons = comparisons
        state.logs.append(f"Completed comparison for {len(comparisons)} companies")
        
        return state

    async def _gap_analysis(self, state: ResearchState) -> ResearchState:
        """Step 6b: Analyze gaps and future trends"""
        self.logger.info("Starting gap analysis...")
        
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
        chain = gap_prompt | self.llm_creative | parser
        
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
            self.logger.error(f"Error during gap analysis: {e}")
            state.strategic_insights = {}
            
        return state

    async def _write_output(self, state: ResearchState) -> ResearchState:
        """Step 7: Write structured output and logs"""
        self.logger.info("Writing output files...")
        
        # Create run directory
        run_dir = self.results_manager.create_run_directory()
        
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
        
        # Write JSON output
        output_file = run_dir / "analysis.json"
        self.results_manager.save_json(output_data, output_file)
        
        # Write execution log
        log_file = run_dir / "execution.log"
        log_content = f"Research Analysis Execution Log\n{'='*50}\n\n" + "\n".join(state.logs)
        self.results_manager.save_text(log_content, log_file)
        
        # Write summary report
        summary_file = run_dir / "summary_report.md"
        await self._generate_summary_report(state, summary_file)
        
        # Generate PDF Report
        try:
            pdf_gen = ReportGenerator()
            pdf_file = run_dir / "research_report.pdf"
            pdf_gen.generate_report(output_data, pdf_file)
            state.logs.append(f"PDF Report generated: {pdf_file}")
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            state.logs.append(f"PDF generation failed: {e}")
        
        state.logs.append(f"Output written to: {run_dir}")
        self.logger.info(f"Analysis complete. Results saved to {run_dir}")
        
        # Save output directory in state for return value
        state.logs.append(f"OUTPUT_DIR:{run_dir}") 

        return state

    async def _generate_summary_report(self, state: ResearchState, file_path: Path):
        """Generate a human-readable summary report"""
        
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
        
        chain = report_prompt | self.llm_creative | StrOutputParser()
        
        try:
            report = await chain.ainvoke({
                "topics": state.topics,
                "total_papers": sum(len(papers) for papers in state.analyzed_papers.values()),
                "top_companies": [comp.company for comp in state.company_comparisons[:5]],
                "topic_stats": [f"{stat.topic}: {stat.paper_count} papers" for stat in state.topic_stats],
                "company_rankings": [f"{comp.ranking}. {comp.company}" for comp in state.company_comparisons[:10]]
            })
            
            self.results_manager.save_text(report, file_path)
                
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")

    async def analyze_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Main entry point for the analysis system"""
        self.logger.info(f"Starting analysis for topics: {topics}")
        
        # Initialize state
        initial_state = ResearchState(topics=topics)
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Extract output directory from logs if available
        output_dir = str(self.results_manager.base_dir)
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
