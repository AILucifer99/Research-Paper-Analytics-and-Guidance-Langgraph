from fpdf import FPDF
from pathlib import Path
from typing import Dict, Any, List

class PDFReport(FPDF):
    def header(self):
        # Logo or Title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Research Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 1, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.safe_multi_cell(body)
        self.ln()
    
    def safe_multi_cell(self, text):
        """Helper to safely render multi-line text"""
        try:
            # Sanitize text
            safe_text = str(text).encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_text)
        except Exception as e:
            # Fallback for errors (e.g., margins, wrapping)
            print(f"PDF Render Warning: {e}")
            self.cell(0, 5, "[Content could not be rendered due to formatting issues]")
            self.ln()

    def add_section(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

class ReportGenerator:
    def __init__(self):
        pass

    def generate_report(self, state_data: Dict[str, Any], filepath: Path):
        pdf = PDFReport()
        pdf.alias_nb_pages()
        
        # Title Page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 60, '', 0, 1)
        pdf.cell(0, 20, "Research Analysis Report", 0, 1, 'C')
        pdf.set_font('Arial', '', 14)
        pdf.cell(0, 10, f"Generated on: {state_data['metadata']['timestamp']}", 0, 1, 'C')
        
        # Topics
        pdf.add_page()
        pdf.chapter_title("Topics Analyzed")
        for topic in state_data['topics']:
            pdf.cell(0, 10, f"- {topic}", 0, 1)
        pdf.ln(5)
            
        # Strategic Insights (Gap Analysis)
        if 'strategic_insights' in state_data['metadata'] and state_data['metadata']['strategic_insights']:
            pdf.add_page()
            pdf.chapter_title("Strategic Insights & Gap Analysis")
            insights = state_data['metadata']['strategic_insights']
            
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, "Identified Gaps:", 0, 1)
            pdf.set_font('Arial', '', 10)
            for gap in insights.get('gaps', []):
                 pdf.safe_multi_cell(f"- {gap}")
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, "Emerging Trends:", 0, 1)
            pdf.set_font('Arial', '', 10)
            for trend in insights.get('trends', []):
                 pdf.safe_multi_cell(f"- {trend}")
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, "Recommendations:", 0, 1)
            pdf.set_font('Arial', '', 10)
            for rec in insights.get('recommendations', []):
                 pdf.safe_multi_cell(f"- {rec}")
        
        # Top Companies
        pdf.add_page()
        pdf.chapter_title("Top Companies & Institutions")
        
        pdf.set_font('Arial', 'B', 10)
        # Simple table header
        pdf.cell(80, 10, "Company", 1)
        pdf.cell(30, 10, "Papers", 1)
        pdf.cell(30, 10, "Innov. Score", 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        for comp in state_data['company_comparisons']:
            pdf.cell(80, 10, comp['company'][:35], 1) # Truncate long names
            pdf.cell(30, 10, str(comp['paper_count']), 1)
            pdf.cell(30, 10, f"{comp['innovation_score']:.2f}", 1)
            pdf.ln()
            
        pdf.output(str(filepath))
