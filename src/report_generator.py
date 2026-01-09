
from pathlib import Path
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.time_style = ParagraphStyle(
            'TimeStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=24
        )
        
        self.section_header = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue
        )
        
        self.sub_header = ParagraphStyle(
            'SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=10,
            textColor=colors.black
        )

    def generate_report(self, state_data: Dict[str, Any], filepath: Path, charts_dir: Optional[Path] = None):
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # --- Title Page ---
        story.append(Spacer(1, 2 * 72))  # Add some vertical space
        
        title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Title'],
            fontSize=28,
            spaceAfter=24,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("Research Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {state_data['metadata']['timestamp']}", self.time_style))
        story.append(PageBreak())
        
        # --- Topics ---
        story.append(Paragraph("Topics Analyzed", self.section_header))
        for topic in state_data['topics']:
            story.append(Paragraph(f"• {topic}", self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # --- Strategic Insights ---
        if 'strategic_insights' in state_data['metadata'] and state_data['metadata']['strategic_insights']:
            story.append(Paragraph("Strategic Insights & Gap Analysis", self.section_header))
            insights = state_data['metadata']['strategic_insights']
            
            if insights.get('gaps'):
                story.append(Paragraph("Identified Gaps", self.sub_header))
                for gap in insights['gaps']:
                    story.append(Paragraph(f"• {gap}", self.styles['BodyText']))
                story.append(Spacer(1, 12))
                
            if insights.get('trends'):
                story.append(Paragraph("Emerging Trends", self.sub_header))
                for trend in insights['trends']:
                    story.append(Paragraph(f"• {trend}", self.styles['BodyText']))
                story.append(Spacer(1, 12))
                
            if insights.get('recommendations'):
                story.append(Paragraph("Recommendations", self.sub_header))
                for rec in insights['recommendations']:
                    story.append(Paragraph(f"• {rec}", self.styles['BodyText']))
                story.append(Spacer(1, 12))
        
        story.append(PageBreak())

        # --- Visual Analytics (Charts) ---
        if charts_dir and charts_dir.exists():
            charts = list(charts_dir.glob("*.png"))
            if charts:
                story.append(Paragraph("Visual Analytics", self.section_header))
                
                for chart in charts:
                    # Clean up filename for title
                    title = chart.stem.replace('_', ' ').title()
                    story.append(Paragraph(title, self.sub_header))
                    
                    # Add image (restraining width to page text width approx 450)
                    img = Image(str(chart), width=450, height=300)
                    img.hAlign = 'CENTER'
                    story.append(img)
                    story.append(Spacer(1, 24))
                
                story.append(PageBreak())
        
        # --- Top Companies Table ---
        story.append(Paragraph("Top Companies & Institutions", self.section_header))
        
        table_data = [['Company', 'Papers', 'Innovation', 'Collaboration']]
        
        for comp in state_data['company_comparisons']:
            table_data.append([
                Paragraph(comp['company'], self.styles['BodyText']), # Use paragraph for wrapping
                str(comp['paper_count']),
                f"{comp['innovation_score']:.2f}",
                f"{comp['collaboration_score']:.2f}"
            ])
            
        # Create Table
        t = Table(table_data, colWidths=[250, 60, 80, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(t)
        
        # Build Document
        doc.build(story)
