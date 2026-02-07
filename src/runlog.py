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


def _cfg_to_jsonable(cfg: Any) -> Any:
    # is_dataclass(...) is True for BOTH dataclass instances and dataclass classes.
    # asdict(...) only accepts instances.
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)

    # Optional support for pydantic models, guarded so type checkers don't complain.
    if not isinstance(cfg, type):
        md = getattr(cfg, "model_dump", None)
        if callable(md):
            return md()

    return str(cfg)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(o)


def write_run_manifest(
    out_dir: Path,
    phase: str,
    cfg: Any,
    extra: Mapping[str, Any] | None = None,
    data_files: list[Path] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"phase{phase}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    env_subset = {
        "AML_DEVICE": str(__import__("os").getenv("AML_DEVICE", "")),
        "AML_FAST": str(__import__("os").getenv("AML_FAST", "")),
        "AML_STRICT_ELLIPTIC": str(__import__("os").getenv("AML_STRICT_ELLIPTIC", "")),
    }

    manifest = {
        "run_id": run_id,
        "phase": phase,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "config": _cfg_to_jsonable(cfg),
        "data_files": [_file_facts(p) for p in (data_files or [])],
        "env": env_subset,
        "extra": dict(extra or {}),
    }
    p = out_dir / f"{run_id}.json"
    p.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return p
