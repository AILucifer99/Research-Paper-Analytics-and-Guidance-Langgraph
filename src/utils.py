from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

class ResultsManager:
    """Handles file operations and result storage"""
    def __init__(self, base_dir: str = "research_output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_run_directory(self) -> Path:
        """Create a unique directory for the current run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        return run_dir
        
    def save_json(self, data: Dict[str, Any], filepath: Path):
        """Save data as JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
    def save_text(self, content: str, filepath: Path):
        """Save text content to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
