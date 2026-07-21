from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AppSettings:
    retrieval_preset: str = "Balanced"
    top_k: str = "Auto"
    neighbors: int = 2
    use_embeddings: bool = True
    show_diagnostics: bool = False
    answer_mode: str = "Strict"
    file_sort: str = "Name"
    geometry: str = "1100x720"
    sidebar_visible: bool = False
    theme: str = "Dark"


def load_app_settings(path: Path) -> AppSettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppSettings()
    if not isinstance(payload, dict):
        return AppSettings()
    defaults = asdict(AppSettings())
    values = {key: payload.get(key, default) for key, default in defaults.items()}
    try:
        return AppSettings(**values)
    except (TypeError, ValueError):
        return AppSettings()


def save_app_settings(path: Path, settings: AppSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
