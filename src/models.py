from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

@dataclass
class Paper:
    title: str
    authors: List[str]
    affiliations: List[str]
    abstract: str
    published: str
    url: str
    categories: List[str]
    summary: str = ""
    contributions: List[str] = None
    methods: List[str] = None
    datasets: List[str] = None
    limitations: List[str] = None
    companies: List[str] = None

@dataclass
class TopicStats:
    topic: str
    paper_count: int
    methods_distribution: Dict[str, int]
    reproducibility_score: float
    open_source_ratio: float
    top_companies: List[str]
    avg_citation_potential: float

@dataclass
class CompanyComparison:
    company: str
    topics_covered: List[str]
    paper_count: int
    strengths: List[str]
    weaknesses: List[str]
    innovation_score: float
    collaboration_score: float
    ranking: int

class ResearchState(BaseModel):
    topics: List[str] = Field(default_factory=list)
    expanded_queries: Dict[str, List[str]] = Field(default_factory=dict)
    papers: Dict[str, List[Paper]] = Field(default_factory=dict)
    analyzed_papers: Dict[str, List[Paper]] = Field(default_factory=dict)
    topic_stats: List[TopicStats] = Field(default_factory=list)
    company_comparisons: List[CompanyComparison] = Field(default_factory=list)
    strategic_insights: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
