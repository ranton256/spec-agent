import json
from pathlib import Path
from typing import List, Dict, Any

class JsonFileMemory:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]")

    def read(self) -> List[Dict[str, Any]]:
        return json.loads(self.file_path.read_text())

    def append(self, entry: Dict[str, Any]):
        content = self.read()
        content.append(entry)
        self.file_path.write_text(json.dumps(content, indent=2))
