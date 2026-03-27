import os
import json
from datetime import datetime
from typing import Optional

from config import Config


class ReportStore:
    """Manages generated report files."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or Config.REPORT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def save_report(self, content: str, title: str, metadata: dict = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)
        safe_title = safe_title.replace(" ", "_")[:50]
        filename = f"{timestamp}_{safe_title}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        if metadata:
            meta_path = filepath.replace(".md", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": title,
                    "created_at": datetime.now().isoformat(),
                    "filename": filename,
                    **metadata,
                }, f, ensure_ascii=False, indent=2)

        return filepath

    def list_reports(self) -> list[dict]:
        reports = []
        for f in sorted(os.listdir(self.output_dir), reverse=True):
            if f.endswith(".md"):
                path = os.path.join(self.output_dir, f)
                meta_path = path.replace(".md", "_meta.json")
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as mf:
                        meta = json.load(mf)
                reports.append({
                    "filename": f,
                    "path": path,
                    "size": os.path.getsize(path),
                    **meta,
                })
        return reports

    def read_report(self, filename: str) -> Optional[str]:
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()
