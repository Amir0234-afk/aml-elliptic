# src/runlog.py
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def _file_facts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }

def _serialize_cfg(cfg: Any) -> Any:
    # Only call asdict on dataclass *instances*, not dataclass classes
    if is_dataclass(cfg) and not isinstance(cfg, type):
        try:
            return asdict(cfg)
        except Exception:
            pass
    # Fallbacks
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return str(cfg)

def write_run_manifest(
    out_dir: Path,
    phase: str,
    cfg: Any,
    extra: Mapping[str, Any] | None = None,
    data_files: list[Path] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"phase{phase}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "run_id": run_id,
        "phase": phase,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "config": _serialize_cfg(cfg),
        "data_files": [_file_facts(p) for p in (data_files or [])],
        "extra": dict(extra or {}),
    }
    p = out_dir / f"{run_id}.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p
