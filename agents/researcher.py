import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

# PDF and text processing
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LLM integration (using OpenAI-compatible API)
from openai import OpenAI


class PDFProcessor:
    """Extracts and processes text from PDF files."""
    
    def __init__(self):
        self.papers_content = {}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def load_papers_from_directory(self, directory: str) -> Dict[str, str]:
        """Load all PDFs from a directory."""
        papers_dir = Path(directory)
        
        for pdf_file in papers_dir.glob("*.pdf"):
            print(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            if text:
                self.papers_content[pdf_file.name] = text
        
        print(f"\nLoaded {len(self.papers_content)} papers")
        return self.papers_content


class CitationManager:
    """Manages citations and references throughout the paper."""
    
    def __init__(self):
        self.citations = {}  # {citation_key: {title, authors, year, source}}
        self.citation_counter = 0
        self.used_citations = []
    
    def extract_citations_from_chunks(self, chunks_metadata: List[Dict]) -> List[Dict]:
        """Extract citation information from retrieved chunks."""
        citations = []
        seen_papers = set()
        
        for chunk in chunks_metadata:
            paper_name = chunk['metadata']['paper']
            if paper_name not in seen_papers:
                seen_papers.add(paper_name)
                
                # Parse paper name (format: arxiv_id_title.pdf)
                citation_key = f"ref{len(citations) + 1}"
                
                citations.append({
                    'key': citation_key,
                    'source': paper_name,
                    'text': chunk['text'][:200]
                })
                
                self.citations[citation_key] = {
                    'source': paper_name,
                    'used': False
                }
        
        return citations
    
    def add_citation(self, text: str, citation_keys: List[str]) -> str:
        """Add citation markers to text."""
        for key in citation_keys:
            if key in self.citations:
                self.citations[key]['used'] = True
                if key not in self.used_citations:
                    self.used_citations.append(key)
        
        citation_str = "[" + ", ".join(citation_keys) + "]"
        return f"{text} {citation_str}"
    
    def generate_references_section(self) -> str:
        """Generate formatted references section."""
        references = "## References\n\n"
        
        for i, key in enumerate(self.used_citations, 1):
            if key in self.citations:
                source = self.citations[key]['source']
                references += f"{i}. {source}\n"
        
        return references


class RAGSystem:
    """Retrieval-Augmented Generation system for paper content."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def build_index(self, papers_content: Dict[str, str]):
        """Build vector index from papers."""
        print("\nBuilding RAG index...")
        
        for paper_name, content in papers_content.items():
            paper_chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(paper_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'paper': paper_name,
                    'chunk_id': i,
                    'text': chunk
                })
        
        # Create embeddings
        print(f"Creating embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        print("✓ Index built successfully")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'score': float(similarities[idx])
            })
        
        return results


class ResearchAgent:
    """Base class for research agents."""
    
    def __init__(self, name: str, role: str, client: OpenAI, model: str = "gpt-4"):
        self.name = name
        self.role = role
        self.client = client
        self.model = model
    
    def generate(self, prompt: str, context: str = "", temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text using LLM."""
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": f"{context}\n\n{prompt}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


class LiteratureReviewAgent(ResearchAgent):
    """Agent specialized in literature review."""
    
    def __init__(self, client: OpenAI, rag_system: RAGSystem, citation_manager: CitationManager):
        super().__init__(
            name="Literature Reviewer",
            role="You are an expert researcher specializing in comprehensive literature reviews. "
                 "You analyze research papers, identify key themes, methodologies, and gaps in existing research. "
                 "You provide proper citations using [ref1], [ref2] format.",
            client=client
        )
        self.rag = rag_system
        self.citation_manager = citation_manager
    
    def review_literature(self, topic: str) -> Tuple[str, List[str]]:
        """Conduct literature review on a topic."""
        print(f"\n[{self.name}] Conducting literature review on: {topic}")
        
        # Retrieve relevant content
        relevant_chunks = self.rag.retrieve(topic, top_k=15)
        
        # Extract citations
        citations = self.citation_manager.extract_citations_from_chunks(relevant_chunks)
        citation_keys = [c['key'] for c in citations]
        
        context = "## Relevant Research Findings:\n\n"
        for i, chunk in enumerate(relevant_chunks[:10], 1):
            ref_key = citations[min(i-1, len(citations)-1)]['key']
            context += f"### Source [{ref_key}] (from {chunk['metadata']['paper']}):\n"
            context += f"{chunk['text'][:600]}...\n\n"
        
        prompt = f"""Based on the provided research papers, write a comprehensive literature review on: {topic}
            Your review should:
            1. Summarize the current state of research with proper citations [ref1], [ref2], etc.
            2. Identify key methodologies and approaches used in the field
            3. Highlight important findings and trends
            4. Compare and contrast different approaches
            5. Note any gaps or limitations in existing work

            IMPORTANT: Use citation format [ref1], [ref2] etc. to reference the sources provided above.

            Write in an academic style with clear structure."""
                    
        review = self.generate(prompt, context, max_tokens=3000)
        
        return review, citation_keys


class MathematicalModelingAgent(ResearchAgent):
    """Agent specialized in mathematical modeling and formalization."""
    
    def __init__(self, client: OpenAI, rag_system: RAGSystem):
        super().__init__(
            name="Mathematical Modeler",
            role="You are an expert in mathematical modeling and formalization. "
                 "You create rigorous mathematical frameworks, derive equations, and formalize concepts using LaTeX notation. "
                 "You understand probability theory, optimization, linear algebra, calculus, and statistics.",
            client=client
        )
        self.rag = rag_system
    
    def develop_mathematical_model(self, topic: str, methodology: str) -> str:
        """Develop mathematical model for the research."""
        print(f"\n[{self.name}] Developing mathematical model...")
        
        # Retrieve mathematical content
        relevant_chunks = self.rag.retrieve(f"mathematical model equations {topic}", top_k=10)
        
        context = "## Mathematical Approaches from Literature:\n\n"
        for chunk in relevant_chunks:
            if any(indicator in chunk['text'].lower() for indicator in 
                   ['equation', 'formula', 'theorem', 'lemma', 'proof', '∑', '∫', 'matrix']):
                context += f"{chunk['text'][:500]}...\n\n"
        
        context += f"\n## Methodology Context:\n{methodology[:1000]}"
        
        prompt = f"""
        Develop a rigorous mathematical model for: {topic}
        Your model should include:
        1. **Problem Formalization**
        - Define the problem mathematically
        - Specify variables, parameters, and constraints
        - Use proper mathematical notation (LaTeX format: $x$, $$equation$$)

        2. **Mathematical Framework**
        - Define the mathematical space and structures
        - Specify relevant mathematical objects (vectors, matrices, functions, etc.)
        - State assumptions clearly

        3. **Core Equations**
        - Derive key equations step-by-step
        - Show mathematical relationships with help of Latex notations

        4. **Optimization Formulation** (if applicable)
        - Define objective function
        - Specify constraints
        - Describe solution approach

        5. **Theoretical Properties**
        - Convergence properties
        - Complexity analysis
        - Theoretical guarantees

        Use standard LaTeX notation for all mathematical expressions.
        - Inline math should be wrapped in single dollars: $x^2$
        - Block equations should be wrapped in double dollars: $$E = mc^2$$
        - Do NOT use markdown code blocks (like ```latex) for equations. 
        - Ensure all symbols are properly defined.
        
        Example:
        "The objective function is defined as:
        $$J(\\theta) = -\\frac{{1}}{{m}} \\sum_{{i=1}}^m [y^{{(i)}} \\log h_\\theta(x^{{(i)}}) + (1-y^{{(i)}}) \\log (1-h_\\theta(x^{{(i)}}))]$$
        where $h_\\theta(x)$ is the hypothesis function."
        """
        
        return self.generate(prompt, context, temperature=0.5, max_tokens=3000)


class MethodologyAgent(ResearchAgent):
    """Agent specialized in methodology design."""
    
    def __init__(self, client: OpenAI, rag_system: RAGSystem, citation_manager: CitationManager):
        super().__init__(
            name="Methodology Expert",
            role="You are an expert in research methodology. You design robust, innovative research "
                 "methodologies based on existing approaches and identify ways to improve upon them. "
                 "You provide proper citations using [ref1], [ref2] format.",
            client=client
        )
        self.rag = rag_system
        self.citation_manager = citation_manager
    
    def design_methodology(self, topic: str, literature_review: str) -> Tuple[str, List[str]]:
        """Design research methodology."""
        print(f"\n[{self.name}] Designing methodology...")
        
        # Retrieve methodology-related content
        relevant_chunks = self.rag.retrieve(f"methodology approach algorithm {topic}", top_k=12)
        
        citations = self.citation_manager.extract_citations_from_chunks(relevant_chunks)
        citation_keys = [c['key'] for c in citations]
        
        context = "## Existing Methodologies:\n\n"
        for i, chunk in enumerate(relevant_chunks[:8], 1):
            ref_key = citations[min(i-1, len(citations)-1)]['key']
            context += f"[{ref_key}]: {chunk['text'][:500]}...\n\n"
        
        context += f"\n## Literature Review Summary:\n{literature_review[:1200]}"
        
        prompt = f"""Based on the existing methodologies and literature review, propose a novel research methodology for: {topic}

            Your methodology should include:

            1. **Research Design and Approach**
            - Overall framework and pipeline
            - Why this approach is suitable
            - Cite relevant methodologies [ref1], [ref2]

            2. **Algorithm/Procedure**
            - Step-by-step description
            - Pseudocode or flowchart description
            - Novel contributions

            3. **Data Collection/Generation Methods**
            - Data sources and requirements
            - Preprocessing steps
            - Quality assurance

            4. **Implementation Details**
            - Technical specifications
            - Tools and frameworks
            - Computational requirements

            5. **Evaluation Metrics**
            - Quantitative metrics
            - Qualitative assessments
            - Comparison baselines

            6. **Limitations and Considerations**
            - Known limitations
            - Ethical considerations
            - Reproducibility

            Use citations [ref1], [ref2] to reference existing work. Be specific and technically sound.
            """
        
        methodology = self.generate(prompt, context, max_tokens=3000)
        
        return methodology, citation_keys


class VisualizationAgent(ResearchAgent):
    """Agent specialized in generating figure and table descriptions."""
    
    def __init__(self, client: OpenAI):
        super().__init__(
            name="Visualization Expert",
            role="You are an expert in data visualization and scientific illustration. "
                 "You design figures, tables, and visual representations that effectively communicate research findings.",
            client=client
        )
    
    def design_figures_and_tables(self, topic: str, sections: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Design figures and tables for the paper."""
        print(f"\n[{self.name}] Designing figures and tables...")
        
        context = f"""
        Methodology: {sections.get('methodology', '')[:1000]}
        Mathematical Model: {sections.get('math_model', '')[:1000]}
        Results: {sections.get('results', '')[:1000]}
        """
        
        prompt = f"""Design a comprehensive set of figures and tables for a research paper on: {topic}

            For each figure/table, provide:
            1. **ID**: Figure/Table number
            2. **Type**: (e.g., "Architecture Diagram", "Performance Comparison", "Algorithm Flowchart", "Results Table")
            3. **Title**: Descriptive title
            4. **Description**: Detailed description of what should be shown
            5. **Caption**: Academic-style caption
            6. **Data/Content**: What specific data or information to display
            7. **Placement**: Which section it belongs to

            Provide 5-7 figures and 2-3 tables that would best support this research.

            Format as JSON:
            ```json
            {{
            "figures": [
                {{
                "id": "fig1",
                "type": "System Architecture",
                "title": "...",
                "description": "...",
                "caption": "Figure 1: ...",
                "section": "Methodology"
                }}
            ],
            "tables": [...]
            }}
            ```
            """
        
        response = self.generate(prompt, context, temperature=0.6, max_tokens=2500)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                visualizations = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                visualizations = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            visualizations = {
                "figures": [
                    {
                        "id": "fig1",
                        "type": "Placeholder",
                        "title": "System Overview",
                        "description": "Overview of the proposed system",
                        "caption": "Figure 1: System overview",
                        "section": "Methodology"
                    }
                ],
                "tables": []
            }
        
        return visualizations


class ResultsAnalysisAgent(ResearchAgent):
    """Agent specialized in results analysis."""
    
    def __init__(self, client: OpenAI, rag_system: RAGSystem, citation_manager: CitationManager):
        super().__init__(
            name="Results Analyst",
            role="You are an expert in analyzing research results and drawing meaningful conclusions. "
                 "You identify patterns, compare findings with existing work, provide statistical insights, "
                 "and use proper citations [ref1], [ref2] format.",
            client=client
        )
        self.rag = rag_system
        self.citation_manager = citation_manager
    
    def analyze_results(self, topic: str, methodology: str, math_model: str, 
                       visualizations: Dict) -> Tuple[str, List[str]]:
        """Analyze hypothetical results."""
        print(f"\n[{self.name}] Analyzing expected results...")
        
        relevant_chunks = self.rag.retrieve(f"results findings performance {topic}", top_k=12)
        
        citations = self.citation_manager.extract_citations_from_chunks(relevant_chunks)
        citation_keys = [c['key'] for c in citations]
        
        context = "## Results from Similar Studies:\n\n"
        for i, chunk in enumerate(relevant_chunks[:8], 1):
            ref_key = citations[min(i-1, len(citations)-1)]['key']
            context += f"[{ref_key}]: {chunk['text'][:500]}...\n\n"
        
        context += f"\n## Methodology:\n{methodology[:800]}"
        context += f"\n## Mathematical Model:\n{math_model[:800]}"
        
        # Add figure references
        fig_refs = "\n## Available Figures:\n"
        for fig in visualizations.get('figures', []):
            fig_refs += f"- {fig['id']}: {fig['title']}\n"
        for table in visualizations.get('tables', []):
            fig_refs += f"- {table['id']}: {table['title']}\n"
        
        context += fig_refs
        
        prompt = f"""Based on similar studies and the proposed methodology, discuss expected results and their analysis for: {topic}

Your analysis should include:

1. **Experimental Setup**
   - Describe datasets/experiments
   - Parameters and configurations
   - Baseline comparisons

2. **Quantitative Results**
   - Present key metrics and measurements
   - Reference figures/tables (e.g., "as shown in Figure 1", "Table 1 presents...")
   - Statistical significance
   - Compare with baselines [ref1], [ref2]

3. **Qualitative Analysis**
   - Interpret the results
   - Discuss patterns and trends
   - Unexpected findings

4. **Ablation Studies** (if applicable)
   - Component-wise analysis
   - Impact of different design choices

5. **Comparison with State-of-the-art**
   - How results compare to existing work [ref1], [ref2]
   - Advantages and improvements
   - Where approach falls short

Reference the figures and tables designed for the paper. Use citations [ref1], [ref2] to compare with existing work.
Be analytical and provide concrete numbers where appropriate."""
        
        results = self.generate(prompt, context, max_tokens=3000)
        
        return results, citation_keys


class PeerReviewAgent(ResearchAgent):
    """Agent that performs critical peer review of the paper."""
    
    def __init__(self, client: OpenAI):
        super().__init__(
            name="Peer Reviewer",
            role="You are a critical peer reviewer for top-tier academic conferences. "
                 "You evaluate papers for scientific rigor, novelty, clarity, and contribution. "
                 "You provide constructive feedback and identify weaknesses.",
            client=client
        )
    
    def review_paper(self, paper_content: Dict[str, str]) -> Dict[str, any]:
        """Perform comprehensive peer review."""
        print(f"\n[{self.name}] Conducting peer review...")
        
        full_paper = f"""
        ABSTRACT: {paper_content.get('abstract', '')[:500]}
        INTRODUCTION: {paper_content.get('introduction', '')[:800]}
        LITERATURE REVIEW: {paper_content.get('literature_review', '')[:800]}
        METHODOLOGY: {paper_content.get('methodology', '')[:800]}
        MATHEMATICAL MODEL: {paper_content.get('math_model', '')[:800]}
        RESULTS: {paper_content.get('results', '')[:800]}
        DISCUSSION: {paper_content.get('discussion', '')[:600]}
        CONCLUSION: {paper_content.get('conclusion', '')[:600]}
        """
        
        prompt = """Perform a comprehensive peer review of this research paper.

Evaluate on these criteria (score 1-5 for each):

1. **Novelty and Originality** (1-5)
   - Is the contribution novel?
   - Does it advance the field?

2. **Technical Quality** (1-5)
   - Is the methodology sound?
   - Are the mathematical formulations rigorous?

3. **Clarity and Presentation** (1-5)
   - Is it well-written and organized?
   - Are concepts explained clearly?

4. **Experimental Validation** (1-5)
   - Are results comprehensive?
   - Are comparisons fair?

5. **Significance** (1-5)
   - Will this impact the field?
   - Is it of broad interest?

Provide:
- Overall score (average)
- Strengths (3-5 points)
- Weaknesses (3-5 points)
- Detailed feedback for each section
- Recommendation: Accept/Minor Revisions/Major Revisions/Reject
- Suggestions for improvement

Format as JSON:
```json
{
  "scores": {
    "novelty": 4,
    "technical_quality": 4,
    "clarity": 3,
    "experimental": 4,
    "significance": 4,
    "overall": 3.8
  },
  "recommendation": "Minor Revisions",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "detailed_feedback": {
    "abstract": "...",
    "methodology": "...",
    "results": "..."
  },
  "suggestions": ["...", "...", "..."]
}
```
"""
        
        review = self.generate(prompt, full_paper, temperature=0.4, max_tokens=2500)
        
        # Parse JSON
        try:
            json_match = re.search(r'```json\n(.*?)\n```', review, re.DOTALL)
            if json_match:
                review_data = json.loads(json_match.group(1))
            else:
                review_data = json.loads(review)
        except:
            review_data = {
                "scores": {"overall": 3.5},
                "recommendation": "Minor Revisions",
                "strengths": ["Good methodology", "Clear presentation"],
                "weaknesses": ["Needs more experiments"],
                "suggestions": ["Add more baselines"]
            }
        
        return review_data


class WritingAgent(ResearchAgent):
    """Agent specialized in academic writing."""
    
    def __init__(self, client: OpenAI):
        super().__init__(
            name="Academic Writer",
            role="You are an expert academic writer. You synthesize research content into clear, "
                 "well-structured academic papers with proper flow, coherence, and academic tone.",
            client=client
        )
    
    def write_section(self, section_name: str, content: str, additional_context: str = "") -> str:
        """Write a specific section of the paper."""
        print(f"\n[{self.name}] Writing {section_name} section...")
        
        prompt = f"""Refine and improve the following {section_name} section for an academic research paper.

{additional_context}

Current content:
{content}

Requirements:
1. Maintain academic tone and rigor
2. Ensure logical flow and coherence
3. Use proper academic language
4. Keep technical accuracy
5. Add transitions where needed
6. Preserve all citations [ref1], [ref2] exactly as they appear
7. Preserve all mathematical formulas and LaTeX notation (DO NOT modify $...$ or $$...$$ blocks)
8. Reference figures/tables appropriately

Output the refined section:"""
        
        return self.generate(prompt, "", max_tokens=2500)
    
    def write_abstract(self, paper_content: Dict[str, str]) -> str:
        """Write abstract based on full paper."""
        print(f"\n[{self.name}] Writing abstract...")
        
        context = f"""
Introduction: {paper_content['introduction'][:600]}...
Methodology: {paper_content['methodology'][:600]}...
Mathematical Model: {paper_content.get('math_model', '')[:400]}...
Results: {paper_content['results'][:600]}...
Conclusion: {paper_content['conclusion'][:600]}...
"""
        
        prompt = """Write a concise academic abstract (200-250 words) that summarizes:
1. Research motivation and problem statement
2. Proposed approach and methodology
3. Key mathematical contributions
4. Main findings and results
5. Significance and impact

Follow standard abstract structure. Be precise and compelling."""
        
        return self.generate(prompt, context, max_tokens=500)
    
    def improve_with_review_feedback(self, section_name: str, content: str, 
                                    feedback: Dict) -> str:
        """Improve section based on peer review feedback."""
        print(f"\n[{self.name}] Improving {section_name} based on feedback...")
        
        relevant_feedback = feedback.get('detailed_feedback', {}).get(section_name.lower(), '')
        weaknesses = feedback.get('weaknesses', [])
        suggestions = feedback.get('suggestions', [])
        
        context = f"""
Peer Review Feedback:
- Specific feedback: {relevant_feedback}
- Weaknesses: {', '.join(weaknesses[:3])}
- Suggestions: {', '.join(suggestions[:3])}

Current content:
{content}
"""
        
        prompt = f"""Improve the {section_name} section based on peer review feedback.

Address the weaknesses and incorporate suggestions while:
1. Maintaining all technical content
2. Improving clarity and presentation
3. Adding missing details
4. Strengthening arguments
5. Preserving citations and formulas

Output the improved section:"""
        
        return self.generate(prompt, context, max_tokens=2500)


class CoordinatorAgent(ResearchAgent):
    """Agent that coordinates the entire research paper writing process."""
    
    def __init__(self, client: OpenAI):
        super().__init__(
            name="Research Coordinator",
            role="You are a research coordinator who oversees the entire research paper creation process.",
            client=client
        )
    
    def create_outline(self, topic: str) -> Dict[str, str]:
        """Create paper outline."""
        print(f"\n[{self.name}] Creating paper outline...")
        
        prompt = f"""Create a detailed outline for a research paper on: {topic}

Include these sections:
1. Title
2. Abstract (placeholder)
3. Introduction
4. Literature Review and Related Work
5. Problem Formulation
6. Mathematical Model
7. Methodology
8. Experimental Setup
9. Results and Analysis
10. Discussion
11. Conclusion and Future Work
12. References

For each section, provide 2-3 bullet points of what should be covered."""
        
        outline = self.generate(prompt, "")
        return {"outline": outline, "topic": topic}


class MultiAgentResearchWriter:
    """Orchestrates multiple agents to write a complete research paper."""
    
    def __init__(self, api_key: str, papers_directory: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Initialize PDF processor and RAG system
        print("="*80)
        print("INITIALIZING MULTI-AGENT RESEARCH WRITING SYSTEM")
        print("="*80)
        
        self.pdf_processor = PDFProcessor()
        papers_content = self.pdf_processor.load_papers_from_directory(papers_directory)
        
        self.rag_system = RAGSystem()
        self.rag_system.build_index(papers_content)
        
        self.citation_manager = CitationManager()
        
        # Initialize agents
        print("\nInitializing specialized agents...")
        self.coordinator = CoordinatorAgent(self.client)
        self.literature_agent = LiteratureReviewAgent(self.client, self.rag_system, self.citation_manager)
        self.math_agent = MathematicalModelingAgent(self.client, self.rag_system)
        self.methodology_agent = MethodologyAgent(self.client, self.rag_system, self.citation_manager)
        self.visualization_agent = VisualizationAgent(self.client)
        self.results_agent = ResultsAnalysisAgent(self.client, self.rag_system, self.citation_manager)
        self.peer_reviewer = PeerReviewAgent(self.client)
        self.writing_agent = WritingAgent(self.client)
        print("✓ All agents initialized")
    
    def write_research_paper(self, topic: str, output_file: str = "research_paper.md",
                           review_iterations: int = 1) -> str:
        """Generate a complete research paper with peer review iterations."""
        print("\n" + "="*80)
        print(f"WRITING RESEARCH PAPER ON: {topic}")
        print("="*80)
        
        # Step 1: Create outline
        outline_data = self.coordinator.create_outline(topic)
        
        # Step 2: Literature Review
        literature_review, lit_citations = self.literature_agent.review_literature(topic)
        
        # Step 3: Mathematical Model
        math_model = self.math_agent.develop_mathematical_model(topic, literature_review)
        
        # Step 4: Methodology
        methodology, method_citations = self.methodology_agent.design_methodology(
            topic, literature_review
        )
        
        # Step 5: Design Figures and Tables
        paper_sections = {
            'methodology': methodology,
            'math_model': math_model,
            'literature_review': literature_review
        }
        visualizations = self.visualization_agent.design_figures_and_tables(topic, paper_sections)
        
        # Step 6: Results Analysis
        results, results_citations = self.results_agent.analyze_results(
            topic, methodology, math_model, visualizations
        )
        
        # Step 7: Write Introduction
        intro_context = f"Literature Review:\n{literature_review[:1000]}\n\nMath Model:\n{math_model[:600]}"
        introduction = self.writing_agent.write_section(
            "Introduction",
            f"Research topic: {topic}\n\nProvide context, motivation, problem statement, and research objectives.",
            intro_context
        )
        
        # Step 8: Write Discussion
        discussion_context = f"Results:\n{results[:1000]}\n\nLiterature:\n{literature_review[:600]}"
        discussion = self.writing_agent.write_section(
            "Discussion",
            "Discuss implications of findings, compare with existing work, address limitations, and broader impact.",
            discussion_context
        )
        
        # Step 9: Write Conclusion
        conclusion = self.writing_agent.write_section(
            "Conclusion",
            f"Summarize the research on {topic}, key contributions, future directions, and impact.",
            f"Introduction:\n{introduction[:600]}\n\nResults:\n{results[:600]}"
        )
        
        # Step 10: Compile paper content
        paper_content = {
            'introduction': introduction,
            'literature_review': literature_review,
            'math_model': math_model,
            'methodology': methodology,
            'results': results,
            'discussion': discussion,
            'conclusion': conclusion
        }
        
        # Step 11: Peer Review
        print("\n" + "="*80)
        print("PEER REVIEW PHASE")
        print("="*80)
        review_feedback = self.peer_reviewer.review_paper(paper_content)
        self._print_review_summary(review_feedback)
        
        # Step 12: Improve based on feedback (if iterations > 0)
        if review_iterations > 0:
            print("\n" + "="*80)
            print(f"IMPROVEMENT PHASE (Iteration 1/{review_iterations})")
            print("="*80)
            
            for section in ['introduction', 'methodology', 'results', 'discussion', 'conclusion']:
                if section in paper_content:
                    improved = self.writing_agent.improve_with_review_feedback(
                        section, paper_content[section], review_feedback
                    )
                    paper_content[section] = improved
        
        # Step 13: Write Abstract (after improvements)
        abstract = self.writing_agent.write_abstract(paper_content)
        
        # Step 14: Generate References
        references = self.citation_manager.generate_references_section()
        
        # Step 15: Compile final paper with visualizations
        final_paper = self._compile_paper(
            topic, abstract, paper_content, visualizations, 
            references, review_feedback
        )
        
        # Save to structured directory
        sanitized_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory structure: research_papers/{topic}/{timestamp}/
        output_dir = Path("research_papers") / sanitized_topic / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update output paths
        paper_path = output_dir / "research_paper.md"
        review_path = output_dir / "peer_review.json"
        
        self._save_paper(final_paper, str(paper_path))
        self._save_review_report(review_feedback, str(review_path))
        
        print("\n" + "="*80)
        print(f"✓ RESEARCH PAPER COMPLETED")
        print(f"  - Paper: {paper_path}")
        print(f"  - Review: {review_path}")
        print("="*80)
        
        return final_paper
    
    def _print_review_summary(self, review: Dict):
        """Print summary of peer review."""
        print(f"\n📊 Overall Score: {review['scores']['overall']}/5.0")
        print(f"🎯 Recommendation: {review['recommendation']}")
        
        print("\n✅ Strengths:")
        for i, strength in enumerate(review['strengths'], 1):
            print(f"  {i}. {strength}")
        
        print("\n⚠️  Weaknesses:")
        for i, weakness in enumerate(review['weaknesses'], 1):
            print(f"  {i}. {weakness}")
        
        print("\n💡 Suggestions:")
        for i, suggestion in enumerate(review['suggestions'][:3], 1):
            print(f"  {i}. {suggestion}")
    
    def _compile_paper(self, topic: str, abstract: str, sections: Dict[str, str],
                      visualizations: Dict, references: str, review: Dict) -> str:
        """Compile all sections into final paper."""
        
        # Insert figure placeholders
        figures_section = "\n\n## Figures\n\n"
        for fig in visualizations.get('figures', []):
            figures_section += f"""
### {fig['caption']}

**Type:** {fig['type']}

**Description:** {fig['description']}

*[Figure placeholder - {fig['id']}]*

---
"""
        
        tables_section = "\n\n## Tables\n\n"
        for table in visualizations.get('tables', []):
            tables_section += f"""
### {table['caption']}

**Description:** {table['description']}

*[Table placeholder - {table['id']}]*

---
"""
        
        paper = f"""# {topic}

*Generated by Multi-Agent RAG Research System*  
*Date: {datetime.now().strftime('%Y-%m-%d')}*

---

## Abstract

{abstract}

**Keywords:** [To be filled based on content]

---

## 1. Introduction

{sections['introduction']}

---

## 2. Literature Review and Related Work

{sections['literature_review']}

---

## 3. Problem Formulation

[This section connects the literature review to the mathematical model]

The research problem can be formally stated as follows, building upon the foundations established in the literature review. We aim to develop a rigorous mathematical framework that addresses the identified gaps.

---

## 4. Mathematical Model

{sections['math_model']}

---

## 5. Methodology

{sections['methodology']}

---

## 6. Experimental Setup

[Details about implementation, datasets, hyperparameters, and experimental configuration]

---

## 7. Results and Analysis

{sections['results']}

{figures_section}

{tables_section}

---

## 8. Discussion

{sections['discussion']}

---

## 9. Conclusion and Future Work

{sections['conclusion']}

---

{references}

---

## Appendix A: Peer Review Summary

**Overall Score:** {review['scores']['overall']}/5.0  
**Recommendation:** {review['recommendation']}

**Key Strengths:**
"""
        
        for strength in review['strengths']:
            paper += f"\n- {strength}"
        
        paper += "\n\n**Areas for Improvement:**\n"
        for weakness in review['weaknesses']:
            paper += f"\n- {weakness}"
        
        paper += f"""

---

## Document Metadata

- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Topic:** {topic}
- **System:** Multi-Agent RAG Research Writer
- **Agents Used:** Coordinator, Literature Reviewer, Mathematical Modeler, Methodology Expert, 
  Visualization Designer, Results Analyst, Peer Reviewer, Academic Writer
- **Source Papers:** {len(self.pdf_processor.papers_content)} PDFs analyzed
- **RAG Chunks:** {len(self.rag_system.chunks)} indexed

---

*This paper was automatically generated using a multi-agent system with retrieval-augmented generation. 
While the content is based on real research papers, the specific methodology, results, and claims 
should be verified and validated through actual implementation and experimentation.*
"""
        
        return paper
    
    def _save_paper(self, paper: str, filename: str):
        """Save paper to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(paper)
        print(f"\n✓ Paper saved to: {filename}")
    
    def _save_review_report(self, review: Dict, filename: str):
        """Save peer review report as JSON."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(review, f, indent=2)
        print(f"✓ Review report saved to: {filename}")

