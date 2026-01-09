import asyncio
import os
from src.graph import ResearchAnalysisSystem

async def main():
    system = ResearchAnalysisSystem(
        google_api_key=os.getenv("GOOGLE_API_KEY"), 
        model_name="gemini-2.0-flash"
    )

    results = await system.analyze_topics([
        "Retrieval Augmented Generation on Video Data.",
        "Multi-Agent Reinforcement Learning with memory.",
        "Agentic Computer Vision System for Image Analysis."
    ])
    
    print("\n" + "="*50)
    print("RESEARCH ANALYSIS COMPLETE")
    print("="*50)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import json
    asyncio.run(main())
