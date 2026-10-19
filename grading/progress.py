"""Each learner's private progress: which module checks they've passed.

Stored in .progress.json at the repo root (git-ignored, never committed or
sent anywhere). Grading never fails because of this file.
"""
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = REPO_ROOT / ".progress.json"

REQUIRED_CHECKS = {"04": {"code", "answer"}}
DEFAULT_REQUIRED = {"answer"}

UNREADABLE_NOTE = (
    "Your progress file couldn't be read, so it was set aside as "
    ".progress.json.bak and progress started fresh."
)
UNWRITABLE_NOTE = "Your progress couldn't be saved this time (the progress file isn't writable)."

_MODULE_DIR = re.compile(r"^(\d\d)-(.+)$")


@dataclass(frozen=True)
class RecordResult:
    attempt: int
    module_complete: bool
    note: str = ""


def required_checks(module: str) -> set[str]:
    return REQUIRED_CHECKS.get(module, DEFAULT_REQUIRED)


def _empty() -> dict:
    return {"version": 1, "modules": {}}


def _valid(data) -> bool:
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("modules"), dict):
        return False
    for entry in data["modules"].values():
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("passed_checks"), list)
            and all(isinstance(c, str) for c in entry["passed_checks"])
            and isinstance(entry.get("attempts"), int)
            and (entry.get("completed_on") is None or isinstance(entry["completed_on"], str))
        ):
            return False
    return True


def _load(path: Path) -> tuple[dict, str]:
    """(data, note). An unreadable or wrongly shaped file is moved aside."""
    if not path.exists():
        return _empty(), ""
    try:
        data = json.loads(path.read_text())
        if _valid(data):
            return data, ""
    except (OSError, ValueError):
        pass
    try:
        path.replace(path.with_name(path.name + ".bak"))
    except OSError:
        pass
    return _empty(), UNREADABLE_NOTE


def _save(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def record(module: str, check: str, passed: bool, *, path: Path | None = None,
           today: date | None = None) -> RecordResult:
    """Count one graded attempt at `check` ("answer" or "code") for `module`."""
    path = path or PROGRESS_FILE
    data, note = _load(path)
    entry = data["modules"].setdefault(module, {"passed_checks": [], "attempts": 0, "completed_on": None})
    entry["attempts"] += 1
    if passed and check not in entry["passed_checks"]:
        entry["passed_checks"] = sorted(entry["passed_checks"] + [check])
    complete = required_checks(module) <= set(entry["passed_checks"])
    if complete and entry["completed_on"] is None:
        entry["completed_on"] = (today or date.today()).isoformat()
    try:
        _save(path, data)
    except OSError:
        note = f"{note} {UNWRITABLE_NOTE}".strip()
    return RecordResult(attempt=entry["attempts"], module_complete=complete, note=note)


def completed_modules(*, path: Path | None = None) -> set[str]:
    data, _ = _load(path or PROGRESS_FILE)
    return {m for m, e in data["modules"].items() if required_checks(m) <= set(e["passed_checks"])}


def reset(*, path: Path | None = None) -> None:
    path = path or PROGRESS_FILE
    for p in (path, path.with_name(path.name + ".tmp")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def list_modules(modules_dir: Path | None = None) -> list[str]:
    modules_dir = modules_dir or REPO_ROOT / "modules"
    try:
        entries = list(modules_dir.iterdir())
    except OSError:
        return []
    return sorted(m.group(1) for p in entries if p.is_dir() and (m := _MODULE_DIR.match(p.name)))


def module_info(folder: Path) -> tuple[str | None, str]:
    """(module number, title) for a notebook running in `folder`."""
    match = _MODULE_DIR.match(folder.name)
    if not match:
        return None, "Exercise"
    number, slug = match.groups()
    title = slug.replace("-", " ").title()
    try:
        first_line = (folder / "README.md").read_text().splitlines()[0]
        heading = re.match(rf"^#\s+Module\s+{number}:\s*(.+?)\s*$", first_line)
        if heading:
            title = heading.group(1)
    except (OSError, IndexError):
        pass
    return number, title
