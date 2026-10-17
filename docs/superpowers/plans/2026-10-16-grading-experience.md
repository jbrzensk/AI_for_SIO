# Grading Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the notebooks' plain `PASS`/`NEEDS WORK` printout with a styled HTML result card (per-point checklist, celebration, encouragement, private progress across modules), produced by a single `check()` call.

**Architecture:** `grading.py` becomes a `grading/` folder: `gemini.py` asks Gemini to judge each rubric point, `progress.py` keeps a private `.progress.json`, `card.py` renders self-contained HTML (inline scoped CSS, CSS-only animation, no JavaScript), and `__init__.py` exposes `check()`, `check_code()`, `reset_progress()`. Python, not the model, decides pass/fail by counting met points against `needed`. Every notebook's locked rubric cell becomes `question` / `points` / `needed`.

**Tech Stack:** Python 3.12, `google-genai` (Gemini free tier), `python-dotenv`, `httpx`, IPython display (ships with Jupyter), `nbformat`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-16-grading-experience-design.md`

## Global Constraints

- No new dependencies; `requirements.txt` is unchanged.
- Model `gemini-3.5-flash-lite`, `temperature=0`, automatic function calling disabled, JSON response with a schema; one `generate_content` call per `check()`.
- Python decides the verdict: passed = (number of points met) ≥ `needed`. The model never decides pass/fail as a whole.
- Cards: inline CSS scoped under the `aisio-` class prefix, CSS-only animation, **no JavaScript**, own white background, animations disabled under `prefers-reduced-motion: reduce`.
- Every dynamic string on a card (titles, point names, notes, next step, error text, details) is HTML-escaped.
- All learner-visible strings are fixed in `card.py`, except per-point notes and the next step, which come from the grader.
- Progress lives only in `.progress.json` at the repo root, git-ignored; it never leaves the learner's computer. Grading never fails because of the progress file.
- Notebook import line stays `from grading import ...`; the old `grade()` function is removed.
- Notebook rubric cells stay locked (`"editable": false, "deletable": false`), right after the import cell.
- Notebooks are committed without outputs (the `nbstripout` filter is installed in this clone; never commit run results).
- Run the test suite with `.venv/bin/pytest -q` from the repo root.

## Review Focus

1. Grader notes, next steps, titles, or a learner's exception messages containing HTML (`<script>`, `<img onerror>`, `&`) render as literal text, never as markup. *(test in Task 3)*
2. The progress file can't be written (missing or read-only folder, disk full): the card still shows, with a one-line note; no exception. *(test in Task 2)*
3. `.progress.json` is valid JSON but the wrong shape (hand-edited, older format): treated as unreadable, moved aside, fresh start. *(test in Task 2)*
4. Gemini returns the right point names in a different order: accepted and shown in the rubric's order; duplicate, missing, or unknown names: the "unreadable" card. *(tests in Task 1)*
5. A notebook runs from outside a module folder (working directory is the repo root or `_template`): card titled "Exercise", no progress strip, nothing recorded, no crash. *(tests in Tasks 2 and 4)*

---

## File Structure

```
grading/__init__.py          # new: public API check(), check_code(), reset_progress(), GradingError
grading/gemini.py            # new: judge(), GradingError, PointResult, Judgement
grading/progress.py          # new: record(), completed_modules(), reset(), module_info(), list_modules()
grading/card.py              # new: verdict_card(), code_card(), error_card(), ProgressView
grading.py                   # deleted (replaced by the grading/ folder)
tests/test_grading.py        # deleted (cases move into tests/test_gemini.py)
tests/test_gemini.py         # new
tests/test_progress.py       # new
tests/test_card.py           # new
tests/test_check.py          # new
.gitignore                   # modified: .progress.json, .progress.json.bak
modules/_template/exercise.ipynb, modules/0{1,2,3}-*/exercise.ipynb   # modified (Task 5)
modules/04-coding-practices/exercise.ipynb                            # modified (Task 6)
CONTRIBUTING.md              # modified (Task 7)
```

Between Task 1 and Task 6 the notebooks still import the removed `grade` and won't run; that's expected on this branch and fixed in Tasks 5 and 6.

---

### Task 1: Per-point judging (`grading/gemini.py`)

**Files:**
- Create: `grading/__init__.py`, `grading/gemini.py`, `tests/test_gemini.py`
- Delete: `grading.py`, `tests/test_grading.py`

**Interfaces:**
- Produces:
  - `grading.gemini.GradingError(kind: str, detail: str = "")`, a `RuntimeError` with `.kind` (one of `ERROR_KINDS`) and `.detail`; an unknown kind raises `ValueError`.
  - `grading.gemini.ERROR_KINDS = ("no_key", "invalid_key", "offline", "rate_limit", "unreadable", "service")`
  - `@dataclass(frozen=True) PointResult(name: str, met: bool, note: str)`
  - `@dataclass(frozen=True) Judgement(points: list[PointResult], next_step: str)`; `points` are in the order of the rubric's `points` dict.
  - `grading.gemini.judge(question: str, points: dict[str, str], answer: str) -> Judgement`
  - `grading.gemini._schema(names) -> dict` (the response schema; point `name` is an enum of the rubric names)
  - `grading.gemini._get_client() -> genai.Client` (tests replace it)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gemini.py`:

```python
import json

import httpx
import pytest
from google.genai import errors

import grading.gemini as gemini

POINTS = {
    "Goal": "says what to produce",
    "Context": "names the data",
    "Output": "gives a format",
    "Boundaries": "says what to verify",
}


def reply(points=None, next_step="Add what the model must verify."):
    if points is None:
        points = [
            {"name": "Goal", "met": True, "note": "Clear goal."},
            {"name": "Context", "met": True, "note": "Names the 1959 data."},
            {"name": "Output", "met": True, "note": "PDF and PNG plots."},
            {"name": "Boundaries", "met": False, "note": "Nothing to verify."},
        ]
    return json.dumps({"points": points, "next_step": next_step})


@pytest.fixture
def fake_gemini(monkeypatch):
    """Replace the Gemini client. Call it with text=<reply> or exc=<exception>.
    Returns a dict that captures what was sent."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    sent = {}

    def install(text=None, exc=None):
        class FakeResponse:
            pass

        FakeResponse.text = text

        class FakeModels:
            def generate_content(self, model, contents, config):
                sent.update(model=model, contents=contents, config=config)
                if exc is not None:
                    raise exc
                return FakeResponse()

        class FakeClient:
            models = FakeModels()

        monkeypatch.setattr(gemini, "_get_client", lambda: FakeClient())
        return sent

    return install


def kind_of(callable_):
    with pytest.raises(gemini.GradingError) as excinfo:
        callable_()
    return excinfo.value.kind


def test_missing_key_raises_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "no_key"


def test_judgement_comes_back_in_rubric_order(fake_gemini):
    fake_gemini(text=reply())
    result = gemini.judge("q", POINTS, "a")
    assert [p.name for p in result.points] == list(POINTS)
    assert [p.met for p in result.points] == [True, True, True, False]
    assert result.points[1].note == "Names the 1959 data."
    assert result.next_step == "Add what the model must verify."


def test_points_in_a_different_order_are_reordered(fake_gemini):
    shuffled = [{"name": n, "met": True, "note": n} for n in ("Boundaries", "Goal", "Output", "Context")]
    fake_gemini(text=reply(shuffled))
    assert [p.name for p in gemini.judge("q", POINTS, "a").points] == list(POINTS)


@pytest.mark.parametrize(
    "points",
    [
        [{"name": "Goal", "met": True, "note": ""}],
        [{"name": n, "met": True, "note": ""} for n in ("Goal", "Goal", "Context", "Output", "Boundaries")],
        [{"name": n, "met": True, "note": ""} for n in ("Goal", "Context", "Output", "Scope")],
        [{"name": n, "met": "yes", "note": ""} for n in POINTS],
    ],
    ids=["missing", "duplicate", "unknown-name", "met-not-boolean"],
)
def test_mismatched_or_malformed_points_are_unreadable(fake_gemini, points):
    fake_gemini(text=reply(points))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "unreadable"


@pytest.mark.parametrize(
    "text",
    ["not json", None, json.dumps({"points": []}), json.dumps(["a", "list"])],
    ids=["not-json", "no-text", "no-points", "wrong-type"],
)
def test_unparseable_replies_are_unreadable(fake_gemini, text):
    fake_gemini(text=text)
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "unreadable"


def test_invalid_key_is_reported_with_detail(fake_gemini):
    fake_gemini(exc=errors.ClientError(400, {"error": {
        "code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}))
    with pytest.raises(gemini.GradingError) as excinfo:
        gemini.judge("q", POINTS, "a")
    assert excinfo.value.kind == "invalid_key"
    assert "API key not valid" in excinfo.value.detail


def test_rate_limit_is_reported(fake_gemini):
    fake_gemini(exc=errors.ClientError(429, {"error": {
        "code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "rate_limit"


def test_other_api_errors_are_service_errors(fake_gemini):
    fake_gemini(exc=errors.ServerError(500, {"error": {
        "code": 500, "message": "Internal error", "status": "INTERNAL"}}))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "service"


def test_network_failure_is_offline(fake_gemini):
    fake_gemini(exc=httpx.ConnectError("[Errno 111] Connection refused"))
    assert kind_of(lambda: gemini.judge("q", POINTS, "a")) == "offline"


def test_request_settings(fake_gemini):
    sent = fake_gemini(text=reply())
    gemini.judge("The question", POINTS, "The answer")
    config = sent["config"]
    assert sent["model"] == "gemini-3.5-flash-lite"
    assert config.temperature == 0
    assert config.automatic_function_calling.disable is True
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None


def test_schema_restricts_point_names_to_the_rubric():
    schema = gemini._schema(POINTS)
    assert schema["properties"]["points"]["items"]["properties"]["name"]["enum"] == list(POINTS)
    assert schema["required"] == ["points", "next_step"]


def test_prompt_lists_every_point_and_never_asks_for_a_verdict(fake_gemini):
    sent = fake_gemini(text=reply())
    gemini.judge("The question", POINTS, "The answer")
    for name, description in POINTS.items():
        assert f"- {name}: {description}" in sent["contents"]
    assert "The question" in sent["contents"] and "The answer" in sent["contents"]
    assert "overall" not in sent["contents"].lower()


def test_unknown_error_kind_is_rejected():
    with pytest.raises(ValueError):
        gemini.GradingError("oops")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_gemini.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'grading.gemini'` (the old `grading.py` is a module, not a package).

- [ ] **Step 3: Replace `grading.py` with the package and implement `gemini.py`**

```bash
git rm -q grading.py tests/test_grading.py
mkdir -p grading
```

Create `grading/__init__.py` (temporary; Task 4 replaces it):

```python
"""Grading for the exercise notebooks."""
from grading.gemini import GradingError

__all__ = ["GradingError"]
```

Create `grading/gemini.py`:

```python
"""One Gemini call that judges an answer point by point.

The model judges each rubric point and suggests one next step; it never
decides the overall verdict (grading/__init__.py counts the points).
"""
import json
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

ERROR_KINDS = ("no_key", "invalid_key", "offline", "rate_limit", "unreadable", "service")


class GradingError(RuntimeError):
    """The answer couldn't be graded. `kind` (one of ERROR_KINDS) selects
    the fix shown on the card; `detail` is the underlying message."""

    def __init__(self, kind: str, detail: str = ""):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown GradingError kind: {kind!r}")
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class PointResult:
    name: str
    met: bool
    note: str


@dataclass(frozen=True)
class Judgement:
    points: list[PointResult]  # in the order of the rubric's points
    next_step: str


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GradingError("no_key")
    return genai.Client(api_key=api_key)


def _schema(names) -> dict:
    return {
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(names)},
                        "met": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                    "required": ["name", "met", "note"],
                },
            },
            "next_step": {"type": "string"},
        },
        "required": ["points", "next_step"],
    }


def _prompt(question: str, points: dict[str, str], answer: str) -> str:
    listed = "\n".join(f"- {name}: {description}" for name, description in points.items())
    return (
        "You are grading a research-group member's exercise answer, one point "
        "at a time. Be strict but fair: mark a point met only if the answer "
        "satisfies its substance, not just its wording.\n\n"
        f"Question:\n{question}\n\n"
        f"Points to judge:\n{listed}\n\n"
        f"Learner answer:\n{answer}\n\n"
        "For each point, give met (true or false) and a one-sentence note on "
        "what the answer does, or is missing, for that point. Then give "
        "next_step: one concrete action that would most improve the answer "
        "(if every point is met, the next improvement). Judge every point "
        "exactly once, using exactly these names: " + ", ".join(points) + "."
    )


def _api_error(exc: errors.APIError) -> GradingError:
    code = getattr(exc, "code", None)
    text = str(exc)
    if code == 429:
        return GradingError("rate_limit", text)
    if code in (401, 403) or (code == 400 and "api key" in text.lower()):
        return GradingError("invalid_key", text)
    return GradingError("service", text)


def _parse(text, points: dict[str, str]) -> Judgement:
    try:
        data = json.loads(text)
        by_name = {}
        for item in data["points"]:
            name = item["name"]
            if name in by_name:
                raise ValueError(f"point judged twice: {name}")
            if not isinstance(item["met"], bool):
                raise ValueError(f"met is not a boolean for {name}")
            by_name[name] = PointResult(name=name, met=item["met"], note=str(item["note"]))
        if set(by_name) != set(points):
            raise ValueError(f"expected points {sorted(points)}, got {sorted(by_name)}")
        return Judgement(points=[by_name[name] for name in points], next_step=str(data["next_step"]))
    except (TypeError, KeyError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise GradingError("unreadable", f"{exc}; reply was {text!r}") from exc


def judge(question: str, points: dict[str, str], answer: str) -> Judgement:
    """Ask Gemini to judge `answer` against each of `points` (name -> what
    a met point looks like). Raises GradingError for anything the learner
    can fix or retry."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_prompt(question, points, answer),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_schema(points),
                # Same answer, same judgement: sampling at the default
                # temperature flipped borderline answers between runs.
                temperature=0,
                # No tools are used; leaving this on makes the SDK print a
                # confusing warning into the learner's notebook.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except errors.APIError as exc:
        raise _api_error(exc) from exc
    except httpx.HTTPError as exc:
        raise GradingError("offline", str(exc)) from exc
    return _parse(response.text, points)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_gemini.py -q`
Expected: all pass (19 tests).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all pass; no `tests/test_grading.py` collected.

- [ ] **Step 6: Commit**

```bash
git add grading/__init__.py grading/gemini.py tests/test_gemini.py
git commit -m "Judge exercise answers point by point in a grading package"
```

---

### Task 2: Private progress (`grading/progress.py`)

**Files:**
- Create: `grading/progress.py`, `tests/test_progress.py`
- Modify: `.gitignore` (append two lines)

**Interfaces:**
- Produces:
  - `REPO_ROOT: Path`, `PROGRESS_FILE: Path` (`REPO_ROOT / ".progress.json"`), `UNREADABLE_NOTE: str`, `UNWRITABLE_NOTE: str`
  - `@dataclass(frozen=True) RecordResult(attempt: int, module_complete: bool, note: str = "")`
  - `record(module: str, check: str, passed: bool, *, path: Path | None = None, today: date | None = None) -> RecordResult`; `check` is `"answer"` or `"code"`
  - `completed_modules(*, path: Path | None = None) -> set[str]`
  - `reset(*, path: Path | None = None) -> None`
  - `module_info(folder: Path) -> tuple[str | None, str]` (module number, title)
  - `list_modules(modules_dir: Path | None = None) -> list[str]`
  - `path=None` means "use `PROGRESS_FILE` at call time", so tests can monkeypatch `progress.PROGRESS_FILE`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress.py`:

```python
import json
import subprocess
from datetime import date

import pytest

from grading import progress


@pytest.fixture
def pfile(tmp_path):
    return tmp_path / ".progress.json"


def test_first_record_creates_the_file_and_counts_the_attempt(pfile):
    result = progress.record("02", "answer", False, path=pfile)
    assert result == progress.RecordResult(attempt=1, module_complete=False, note="")
    entry = json.loads(pfile.read_text())["modules"]["02"]
    assert entry == {"passed_checks": [], "attempts": 1, "completed_on": None}


def test_a_pass_completes_the_module_and_stamps_the_date(pfile):
    progress.record("02", "answer", False, path=pfile)
    result = progress.record("02", "answer", True, path=pfile, today=date(2026, 10, 16))
    assert result.attempt == 2 and result.module_complete
    assert json.loads(pfile.read_text())["modules"]["02"]["completed_on"] == "2026-10-16"


def test_passes_stick_after_a_later_not_yet(pfile):
    progress.record("02", "answer", True, path=pfile, today=date(2026, 10, 16))
    result = progress.record("02", "answer", False, path=pfile, today=date(2026, 10, 17))
    assert result.module_complete and result.attempt == 2
    assert json.loads(pfile.read_text())["modules"]["02"]["completed_on"] == "2026-10-16"
    assert progress.completed_modules(path=pfile) == {"02"}


def test_module_04_needs_both_the_code_test_and_the_answer(pfile):
    assert not progress.record("04", "answer", True, path=pfile).module_complete
    assert progress.record("04", "code", True, path=pfile).module_complete
    assert progress.completed_modules(path=pfile) == {"04"}


def test_an_unreadable_file_is_moved_aside_and_progress_starts_fresh(pfile):
    pfile.write_text("{not json")
    result = progress.record("01", "answer", True, path=pfile)
    assert result.note == progress.UNREADABLE_NOTE and result.attempt == 1
    assert pfile.with_name(".progress.json.bak").read_text() == "{not json"
    assert json.loads(pfile.read_text())["modules"]["01"]["attempts"] == 1


def test_valid_json_with_the_wrong_shape_is_treated_as_unreadable(pfile):
    pfile.write_text(json.dumps({"modules": ["01", "02"]}))
    result = progress.record("01", "answer", True, path=pfile)
    assert result.note == progress.UNREADABLE_NOTE
    assert pfile.with_name(".progress.json.bak").exists()


def test_an_unwritable_progress_file_never_breaks_grading(tmp_path):
    unwritable = tmp_path / "no-such-folder" / ".progress.json"
    result = progress.record("02", "answer", True, path=unwritable)
    assert result.module_complete and result.attempt == 1
    assert result.note == progress.UNWRITABLE_NOTE


def test_no_temporary_file_is_left_behind(pfile):
    progress.record("02", "answer", True, path=pfile)
    assert sorted(p.name for p in pfile.parent.iterdir()) == [".progress.json"]


def test_reset_removes_the_file(pfile):
    progress.record("02", "answer", True, path=pfile)
    progress.reset(path=pfile)
    assert not pfile.exists()
    assert progress.completed_modules(path=pfile) == set()


def test_reset_without_a_file_is_fine(pfile):
    progress.reset(path=pfile)


def test_default_path_is_read_at_call_time(monkeypatch, pfile):
    monkeypatch.setattr(progress, "PROGRESS_FILE", pfile)
    progress.record("03", "answer", True)
    assert progress.completed_modules() == {"03"}
    progress.reset()
    assert not pfile.exists()


def test_module_info_reads_the_number_and_title_from_the_readme(tmp_path):
    folder = tmp_path / "02-prompt-engineering"
    folder.mkdir()
    (folder / "README.md").write_text("# Module 02: Prompt Engineering\n\nText\n")
    assert progress.module_info(folder) == ("02", "Prompt Engineering")


def test_module_info_falls_back_to_the_folder_name(tmp_path):
    folder = tmp_path / "05-github-basics"
    folder.mkdir()
    assert progress.module_info(folder) == ("05", "Github Basics")


def test_module_info_outside_a_module_folder(tmp_path):
    assert progress.module_info(tmp_path) == (None, "Exercise")
    assert progress.module_info(tmp_path / "_template") == (None, "Exercise")


def test_list_modules_finds_numbered_folders_only(tmp_path):
    for name in ("02-b", "01-a", "_template", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "03-c.md").write_text("")
    assert progress.list_modules(tmp_path) == ["01", "02"]


def test_list_modules_of_a_missing_folder_is_empty(tmp_path):
    assert progress.list_modules(tmp_path / "nope") == []


def test_list_modules_of_the_real_repo():
    assert progress.list_modules()[:4] == ["01", "02", "03", "04"]


def test_progress_files_are_git_ignored():
    for name in (".progress.json", ".progress.json.bak"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=progress.REPO_ROOT).returncode == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_progress.py -q`
Expected: collection error, `ImportError: cannot import name 'progress' from 'grading'`.

- [ ] **Step 3: Implement `grading/progress.py` and ignore the files**

Create `grading/progress.py`:

```python
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
```

Append to `.gitignore`:

```text
# Each learner's private exercise progress (grading/progress.py).
.progress.json
.progress.json.bak
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_progress.py -q`
Expected: all pass (18 tests).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add grading/progress.py tests/test_progress.py .gitignore
git commit -m "Track each learner's exercise progress privately in .progress.json"
```

---

### Task 3: The HTML card (`grading/card.py`)

**Files:**
- Create: `grading/card.py`, `tests/test_card.py`

**Interfaces:**
- Consumes: `grading.gemini.Judgement`, `grading.gemini.PointResult` (Task 1).
- Produces:
  - `@dataclass(frozen=True) ProgressView(modules: list[str], complete: set[str], current: str | None, note: str = "")`
  - `verdict_card(*, module: str | None, title: str, judgement: Judgement, needed: int, passed: bool, attempt: int | None, module_complete: bool, progress: ProgressView | None) -> str`
  - `code_card(*, module: str | None, title: str, passed: bool, error: str | None, attempt: int | None, module_complete: bool, progress: ProgressView | None) -> str`
  - `error_card(kind: str, detail: str = "") -> str`
  - Confetti on every pass; the "MODULE NN COMPLETE" stamp only when `module` is set and `module_complete` is true (for module 04, a code-test pass alone doesn't complete the module, so it gets confetti without a stamp).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_card.py`:

```python
import re

import pytest

from grading import card
from grading.gemini import Judgement, PointResult

NAMES = ["Goal", "Context", "Output", "Boundaries"]
VIEW = card.ProgressView(modules=["01", "02", "03", "04"], complete={"01", "02"}, current="02")


def judgement(*mets, next_step="Add what the model must verify."):
    return Judgement(
        points=[PointResult(name, met, f"note about {name}") for name, met in zip(NAMES, mets)],
        next_step=next_step,
    )


def pass_card(**overrides):
    args = dict(module="02", title="Prompt Engineering", judgement=judgement(True, True, True, False),
                needed=3, passed=True, attempt=2, module_complete=True, progress=VIEW)
    args.update(overrides)
    return card.verdict_card(**args)


def test_pass_card_has_header_points_next_improvement_and_both_celebrations():
    html = pass_card()
    assert "Passed: Module 02, Prompt Engineering" in html
    assert "3 of 4 points (needed 3) · attempt 2" in html
    assert html.count("✅") == 3 and "➖" in html and "❌" not in html
    assert "Next improvement:" in html
    assert 'class="aisio-confetti"' in html
    assert "MODULE 02<br>COMPLETE" in html


def test_stamp_only_when_the_module_is_complete():
    html = pass_card(module_complete=False)
    assert 'class="aisio-confetti"' in html
    assert "COMPLETE" not in html


def test_not_yet_card():
    html = card.verdict_card(
        module="02", title="Prompt Engineering", judgement=judgement(True, False, False, False), needed=3,
        passed=False, attempt=1, module_complete=False,
        progress=card.ProgressView(["01", "02", "03", "04"], {"01"}, "02"))
    assert "Not yet: Module 02, Prompt Engineering" in html
    assert "1 of 4 points (needed 3) · attempt 1" in html
    assert html.count("❌") == 3
    assert "Your next step:" in html
    assert "Most people need two or three tries." in html
    assert 'class="aisio-confetti"' not in html


def test_progress_strip_marks_done_and_current_modules():
    html = pass_card()
    assert '<span class="aisio-badge aisio-done">01 ✓</span>' in html
    assert '<span class="aisio-badge aisio-done aisio-now">02 ✓</span>' in html
    assert '<span class="aisio-badge">03</span>' in html
    assert "2 of 4 complete" in html


def test_no_progress_strip_or_attempt_outside_a_module():
    html = card.verdict_card(module=None, title="Exercise", judgement=judgement(True), needed=1, passed=True,
                             attempt=None, module_complete=False, progress=None)
    assert "Passed: Exercise" in html
    assert "1 of 1 points (needed 1)" in html and "attempt" not in html
    assert "Your progress" not in html and "COMPLETE" not in html


def test_progress_note_is_shown():
    view = card.ProgressView(["01"], set(), "01", note="Your progress file couldn't be read.")
    html = card.verdict_card(module="01", title="How AI Works", judgement=judgement(False), needed=1,
                             passed=False, attempt=1, module_complete=False, progress=view)
    assert "Your progress file couldn&#x27;t be read." in html


def test_grader_text_is_escaped_never_rendered_as_markup():
    evil = Judgement(points=[PointResult("Goal", False, "<script>alert(1)</script> & <b>bold</b>")],
                     next_step="<img src=x onerror=alert(1)>")
    html = card.verdict_card(module="02", title="Prompt <Engineering>", judgement=evil, needed=1,
                             passed=False, attempt=1, module_complete=False, progress=None)
    assert "<script>" not in html and "<img" not in html and "<b>bold</b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;" in html
    assert "Prompt &lt;Engineering&gt;" in html


def test_cards_contain_no_javascript():
    cards = [
        pass_card(),
        card.error_card("no_key"),
        card.code_card(module="04", title="Coding with AI", passed=False, error="AssertionError: x",
                       attempt=1, module_complete=False, progress=None),
    ]
    for html in cards:
        # no <script> tags and no inline event handlers (onclick=, onerror=, ...)
        assert not re.search(r"<script|\son[a-z]+\s*=", html, re.IGNORECASE)


def test_animations_respect_reduced_motion():
    assert "@media (prefers-reduced-motion: reduce)" in pass_card()


def test_confetti_is_deterministic():
    assert pass_card() == pass_card()


def test_code_card_pass_and_not_yet():
    ok = card.code_card(module="04", title="Coding with AI", passed=True, error=None, attempt=1,
                        module_complete=False, progress=None)
    assert "Code test passed: Module 04, Coding with AI" in ok
    assert "Finish the rest of this exercise below." in ok
    assert 'class="aisio-confetti"' in ok and "COMPLETE" not in ok
    bad = card.code_card(module="04", title="Coding with AI", passed=False,
                         error="AssertionError: Expected a lazy Dask array <before> .compute()",
                         attempt=2, module_complete=False, progress=None)
    assert "Code test not passing yet: Module 04, Coding with AI" in bad
    assert "AssertionError: Expected a lazy Dask array &lt;before&gt; .compute()" in bad
    assert "fix the function, and run the test cell again" in bad


def test_code_card_completing_the_module_shows_the_stamp():
    html = card.code_card(module="04", title="Coding with AI", passed=True, error=None, attempt=3,
                          module_complete=True, progress=None)
    assert "MODULE 04<br>COMPLETE" in html and "this module is complete" in html


@pytest.mark.parametrize("kind, phrase", [
    ("no_key", "Copy <code>.env.example</code> to <code>.env</code>"),
    ("invalid_key", "the API key was rejected"),
    ("offline", "Check your internet connection"),
    ("rate_limit", "resets tomorrow"),
    ("unreadable", "Your answer wasn&#x27;t the problem"),
    ("service", "Google&#x27;s service had a problem"),
])
def test_each_error_kind_shows_its_own_fix(kind, phrase):
    html = card.error_card(kind)
    assert 'class="aisio-card aisio-error"' in html
    assert phrase in html


def test_error_card_detail_is_collapsed_and_escaped():
    html = card.error_card("service", "500 <internal>")
    assert "<details><summary>Technical details</summary><pre>500 &lt;internal&gt;</pre></details>" in html


def test_error_card_without_detail_has_no_details_block():
    assert "<details>" not in card.error_card("no_key")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_card.py -q`
Expected: collection error, `ImportError: cannot import name 'card' from 'grading'`.

- [ ] **Step 3: Implement `grading/card.py`**

```python
"""HTML cards shown under a notebook's grading cell.

One design for every module (spec: 2026-10-16 grading experience). All CSS
is inline and scoped under the aisio- prefix, animation is CSS-only, there
is no JavaScript, and every dynamic string is HTML-escaped.
"""
import random
from dataclasses import dataclass
from html import escape

from grading.gemini import Judgement


@dataclass(frozen=True)
class ProgressView:
    modules: list[str]  # every module number, e.g. ["01", "02", "03", "04"]
    complete: set[str]  # module numbers completed so far
    current: str | None  # the module this notebook belongs to
    note: str = ""  # e.g. the "progress file was set aside" notice


NUDGE = "Most people need two or three tries. Edit your answer above and run this cell again."
_CONFETTI_COLORS = ("#0969da", "#1a7f37", "#8250df", "#bc4c00", "#bf8700", "#cf222e")

_CSS = """<style>
.aisio-card{position:relative;overflow:hidden;max-width:640px;margin:4px 0;background:#ffffff;color:#1f2328;border:1px solid #d1d9e0;border-radius:12px;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;}
.aisio-card .aisio-head{display:flex;align-items:center;gap:12px;padding:14px 18px;}
.aisio-card.aisio-stamped .aisio-head{padding-right:132px;}
.aisio-card .aisio-icon{font-size:28px;line-height:1;}
.aisio-card .aisio-title{font-size:17px;font-weight:700;}
.aisio-card .aisio-sub{font-size:13px;color:#59636e;}
.aisio-card.aisio-pass .aisio-head{background:#dafbe1;border-bottom:1px solid #1a7f37;}
.aisio-card.aisio-pass .aisio-title{color:#1a7f37;}
.aisio-card.aisio-notyet .aisio-head{background:#fff8c5;border-bottom:1px solid #9a6700;}
.aisio-card.aisio-notyet .aisio-title{color:#7d4e00;}
.aisio-card.aisio-error .aisio-head{background:#ddf4ff;border-bottom:1px solid #0969da;}
.aisio-card.aisio-error .aisio-title{color:#0969da;}
.aisio-card .aisio-body{padding:12px 18px 6px;}
.aisio-card ul.aisio-points{list-style:none;margin:0;padding:0;}
.aisio-card ul.aisio-points li{display:grid;grid-template-columns:24px 1fr;gap:6px;padding:5px 0;border-bottom:1px dashed #eaeef2;}
.aisio-card ul.aisio-points li:last-child{border-bottom:0;}
.aisio-card .aisio-pname{font-weight:600;}
.aisio-card .aisio-pnote{color:#59636e;overflow-wrap:anywhere;}
.aisio-card .aisio-next{margin:10px 0 8px;padding:10px 12px;border-radius:8px;background:#f6f8fa;border-left:4px solid #8250df;overflow-wrap:anywhere;}
.aisio-card .aisio-next b{color:#8250df;}
.aisio-card .aisio-nudge,.aisio-card .aisio-note{font-size:13px;color:#59636e;margin:0 0 10px;}
.aisio-card pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0 10px;padding:8px 10px;background:#f6f8fa;border-radius:6px;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#1f2328;}
.aisio-card ol.aisio-fix{margin:6px 0 10px;padding-left:20px;}
.aisio-card code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;background:#eff2f5;padding:1px 5px;border-radius:4px;}
.aisio-card details{margin:0 0 10px;font-size:12px;color:#59636e;}
.aisio-card .aisio-foot{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:10px 18px;border-top:1px solid #eaeef2;background:#f6f8fa;font-size:12px;color:#59636e;}
.aisio-card .aisio-strip{display:flex;gap:6px;flex-wrap:wrap;}
.aisio-card .aisio-badge{min-width:44px;text-align:center;padding:2px 8px;border-radius:999px;border:1px solid #d1d9e0;background:#ffffff;font-weight:600;}
.aisio-card .aisio-badge.aisio-done{background:#1a7f37;border-color:#1a7f37;color:#ffffff;}
.aisio-card .aisio-badge.aisio-now{box-shadow:0 0 0 2px #2da44e66;}
.aisio-card .aisio-confetti{position:absolute;inset:0;pointer-events:none;}
.aisio-card .aisio-confetti i{position:absolute;top:-12px;width:8px;height:12px;border-radius:2px;opacity:0;animation:aisio-fall 2.4s ease-out forwards;}
@keyframes aisio-fall{0%{transform:translateY(0) rotate(0);opacity:1;}100%{transform:translateY(240px) rotate(540deg);opacity:0;}}
.aisio-card .aisio-stamp{position:absolute;right:14px;top:12px;transform:rotate(-12deg) scale(0.4);opacity:0;animation:aisio-stamp 0.6s cubic-bezier(.2,1.6,.4,1) 0.2s forwards;border:3px solid #1a7f37;color:#1a7f37;border-radius:10px;padding:4px 10px;font-weight:800;font-size:13px;letter-spacing:0.04em;background:#ffffffcc;text-align:center;line-height:1.2;}
@keyframes aisio-stamp{to{transform:rotate(-12deg) scale(1);opacity:1;}}
@media (prefers-reduced-motion: reduce){.aisio-card .aisio-confetti{display:none;}.aisio-card .aisio-stamp{animation:none;opacity:1;transform:rotate(-12deg);}}
</style>"""

_ERRORS = {
    "no_key": ("🔑", "Couldn't grade yet: no API key found",
               "Your answer is fine; the grader just needs a key. This takes about 2 minutes.",
               ["Get a free key at <b>aistudio.google.com/apikey</b> (no credit card).",
                "Copy <code>.env.example</code> to <code>.env</code> in the repo folder.",
                "Put your key after <code>GOOGLE_API_KEY=</code> and save.",
                "Restart the kernel and run this cell again."]),
    "invalid_key": ("🔑", "Couldn't grade yet: the API key was rejected",
                    "Your answer is fine; the key in .env isn't working.",
                    ["Open <code>.env</code> in the repo folder and check the line starts exactly "
                     "<code>GOOGLE_API_KEY=</code>, with no spaces or quotes.",
                     "If the key is wrong or was deleted, make a new one at "
                     "<b>aistudio.google.com/apikey</b> and paste it in.",
                     "Restart the kernel and run this cell again."]),
    "offline": ("📡", "Couldn't grade yet: can't reach the grader",
                "Your answer is fine; the grader needs an internet connection.",
                ["Check your internet connection (or VPN).",
                 "Run this cell again."]),
    "rate_limit": ("⏳", "Couldn't grade yet: free-tier limit reached",
                   "Your answer is fine; you've hit Google's free usage limit.",
                   ["Wait a minute and run this cell again.",
                    "If it still says this, you've used today's free allowance; it resets tomorrow."]),
    "unreadable": ("🔧", "Couldn't grade yet: the grader's reply was garbled",
                   "Your answer wasn't the problem.",
                   ["Run this cell again; this is usually a one-off."]),
    "service": ("🔧", "Couldn't grade yet: Google's service had a problem",
                "Your answer is fine.",
                ["Wait a minute and run this cell again."]),
}


def _confetti(seed: str) -> str:
    rng = random.Random(seed)
    pieces = "".join(
        f'<i style="left:{rng.uniform(0, 100):.1f}%;background:{_CONFETTI_COLORS[i % len(_CONFETTI_COLORS)]};'
        f'animation-delay:{rng.uniform(0.1, 0.7):.2f}s"></i>'
        for i in range(28)
    )
    return f'<div class="aisio-confetti">{pieces}</div>'


def _celebration(module: str | None, attempt: int | None, module_complete: bool) -> tuple[str, bool]:
    extra = _confetti(f"{module}-{attempt}")
    if module and module_complete:
        return extra + f'<div class="aisio-stamp">MODULE {escape(module)}<br>COMPLETE</div>', True
    return extra, False


def _strip(progress: ProgressView | None) -> str:
    if progress is None or not progress.modules:
        return ""
    badges = []
    for number in progress.modules:
        classes = ["aisio-badge"]
        if number in progress.complete:
            classes.append("aisio-done")
        if number == progress.current:
            classes.append("aisio-now")
        label = f"{escape(number)} ✓" if number in progress.complete else escape(number)
        badges.append(f'<span class="{" ".join(classes)}">{label}</span>')
    done = sum(1 for m in progress.modules if m in progress.complete)
    return ('<div class="aisio-foot"><span>Your progress</span>'
            f'<span class="aisio-strip">{"".join(badges)}</span>'
            f'<span>{done} of {len(progress.modules)} complete</span></div>')


def _note(progress: ProgressView | None) -> str:
    return f'<p class="aisio-note">{escape(progress.note)}</p>' if progress and progress.note else ""


def _label(module: str | None, title: str) -> str:
    return f"Module {module}, {title}" if module else title


def _card(state: str, icon: str, title: str, sub: str, body: str, *,
          extra: str = "", stamped: bool = False, foot: str = "") -> str:
    classes = f"aisio-card aisio-{state}" + (" aisio-stamped" if stamped else "")
    return (f'{_CSS}<div class="{classes}">{extra}'
            f'<div class="aisio-head"><div class="aisio-icon">{icon}</div>'
            f'<div><div class="aisio-title">{escape(title)}</div>'
            f'<div class="aisio-sub">{escape(sub)}</div></div></div>'
            f'<div class="aisio-body">{body}</div>{foot}</div>')


def verdict_card(*, module: str | None, title: str, judgement: Judgement, needed: int, passed: bool,
                 attempt: int | None, module_complete: bool, progress: ProgressView | None) -> str:
    met = sum(p.met for p in judgement.points)
    sub = f"{met} of {len(judgement.points)} points (needed {needed})"
    if attempt:
        sub += f" · attempt {attempt}"
    miss = "➖" if passed else "❌"
    items = "".join(
        f'<li><span>{"✅" if p.met else miss}</span><span><span class="aisio-pname">{escape(p.name)}:</span> '
        f'<span class="aisio-pnote">{escape(p.note)}</span></span></li>'
        for p in judgement.points
    )
    label = "Next improvement" if passed else "Your next step"
    body = (f'<ul class="aisio-points">{items}</ul>'
            f'<div class="aisio-next"><b>{label}:</b> {escape(judgement.next_step)}</div>'
            + ("" if passed else f'<p class="aisio-nudge">{NUDGE}</p>')
            + _note(progress))
    if passed:
        extra, stamped = _celebration(module, attempt, module_complete)
        return _card("pass", "🎉", f"Passed: {_label(module, title)}", sub, body,
                     extra=extra, stamped=stamped, foot=_strip(progress))
    return _card("notyet", "🌱", f"Not yet: {_label(module, title)}", sub, body, foot=_strip(progress))


def code_card(*, module: str | None, title: str, passed: bool, error: str | None, attempt: int | None,
              module_complete: bool, progress: ProgressView | None) -> str:
    sub = "The test passed" if passed else "The test didn't pass yet"
    if attempt:
        sub += f" · attempt {attempt}"
    if passed:
        nxt = "Nothing left: this module is complete." if module_complete else "Finish the rest of this exercise below."
        body = f'<div class="aisio-next"><b>Next:</b> {nxt}</div>' + _note(progress)
        extra, stamped = _celebration(module, attempt, module_complete)
        return _card("pass", "🎉", f"Code test passed: {_label(module, title)}", sub, body,
                     extra=extra, stamped=stamped, foot=_strip(progress))
    body = (f"<pre>{escape(error or '')}</pre>"
            '<div class="aisio-next"><b>Your next step:</b> read the message above, fix the function, '
            "and run the test cell again.</div>"
            f'<p class="aisio-nudge">{NUDGE}</p>' + _note(progress))
    return _card("notyet", "🌱", f"Code test not passing yet: {_label(module, title)}", sub, body,
                 foot=_strip(progress))


def error_card(kind: str, detail: str = "") -> str:
    icon, title, sub, steps = _ERRORS[kind]
    body = '<ol class="aisio-fix">' + "".join(f"<li>{step}</li>" for step in steps) + "</ol>"
    if detail:
        body += f"<details><summary>Technical details</summary><pre>{escape(detail)}</pre></details>"
    return _card("error", icon, title, sub, body)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_card.py -q`
Expected: all pass (20 tests).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add grading/card.py tests/test_card.py
git commit -m "Render grading results as a self-contained HTML card"
```

---

### Task 4: Public API (`check`, `check_code`, `reset_progress`)

**Files:**
- Modify: `grading/__init__.py` (replace the Task 1 placeholder entirely)
- Create: `tests/test_check.py`

**Interfaces:**
- Consumes: `grading.gemini.judge`, `GradingError`, `Judgement`, `PointResult` (Task 1); `grading.progress.record`, `completed_modules`, `reset`, `module_info`, `list_modules`, `PROGRESS_FILE` (Task 2); `grading.card.verdict_card`, `code_card`, `error_card`, `ProgressView` (Task 3).
- Produces (what notebooks call):
  - `check(question: str, points: dict[str, str], needed: int, answer: str) -> None`
  - `check_code(test_fn: Callable[[], object]) -> None`
  - `reset_progress() -> None`
  - `GradingError` re-exported.
  - `grading._show(html: str) -> None` displays HTML via IPython (tests replace it); `grading.judge` is the name `check()` calls (tests replace it).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_check.py`:

```python
import pytest

import grading
from grading import progress
from grading.gemini import GradingError, Judgement, PointResult

POINTS = {"Goal": "g", "Context": "c", "Output": "o", "Boundaries": "b"}


@pytest.fixture
def shown(monkeypatch):
    cards = []
    monkeypatch.setattr(grading, "_show", cards.append)
    return cards


def make_module(tmp_path, monkeypatch, folder_name, heading):
    folder = tmp_path / folder_name
    folder.mkdir()
    (folder / "README.md").write_text(f"# {heading}\n")
    monkeypatch.chdir(folder)
    pfile = tmp_path / ".progress.json"
    monkeypatch.setattr(progress, "PROGRESS_FILE", pfile)
    return pfile


@pytest.fixture
def module02(tmp_path, monkeypatch):
    return make_module(tmp_path, monkeypatch, "02-prompt-engineering", "Module 02: Prompt Engineering")


def fake_judge(monkeypatch, *mets, exc=None):
    def judge(question, points, answer):
        if exc is not None:
            raise exc
        return Judgement([PointResult(n, m, f"note {n}") for n, m in zip(points, mets)], "next step")

    monkeypatch.setattr(grading, "judge", judge)


def test_pass_shows_the_pass_card_and_records_progress(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, True, True, False)
    assert grading.check("q", POINTS, 3, "my answer") is None
    assert len(shown) == 1
    assert "Passed: Module 02, Prompt Engineering" in shown[0]
    assert "MODULE 02<br>COMPLETE" in shown[0]
    assert progress.completed_modules(path=module02) == {"02"}


def test_not_yet_then_pass_counts_attempts(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, False, False, False)
    grading.check("q", POINTS, 3, "a")
    assert "Not yet: Module 02, Prompt Engineering" in shown[0] and "attempt 1" in shown[0]
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert "Passed: Module 02, Prompt Engineering" in shown[1] and "attempt 2" in shown[1]


def test_progress_strip_includes_the_module_just_completed(monkeypatch, shown, module02):
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert '<span class="aisio-badge aisio-done aisio-now">02 ✓</span>' in shown[0]


def test_grading_errors_show_an_error_card_and_do_not_count(monkeypatch, shown, module02):
    fake_judge(monkeypatch, exc=GradingError("offline", "connection refused"))
    grading.check("q", POINTS, 3, "a")
    assert 'class="aisio-card aisio-error"' in shown[0]
    assert "Check your internet connection" in shown[0]
    assert not module02.exists()


@pytest.mark.parametrize("points, needed", [
    ({}, 1), (POINTS, 0), (POINTS, 5), (POINTS, True), (POINTS, 2.0), (["Goal"], 1), ({"Goal": 1}, 1),
], ids=["empty", "zero", "too-many", "bool", "float", "list", "non-str-description"])
def test_wrong_arguments_raise_value_error(monkeypatch, shown, module02, points, needed):
    fake_judge(monkeypatch, True)
    with pytest.raises(ValueError):
        grading.check("q", points, needed, "a")
    assert shown == []


def test_outside_a_module_folder_nothing_is_recorded(monkeypatch, shown, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(progress, "PROGRESS_FILE", tmp_path / ".progress.json")
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    assert "Passed: Exercise" in shown[0] and "Your progress" not in shown[0]
    assert not (tmp_path / ".progress.json").exists()


def test_check_code_then_explanation_completes_module_04(monkeypatch, shown, tmp_path):
    pfile = make_module(tmp_path, monkeypatch, "04-coding-practices", "Module 04: Coding with AI")

    def failing():
        assert False, "Expected a lazy Dask array before .compute()"

    grading.check_code(failing)
    assert "Code test not passing yet: Module 04, Coding with AI" in shown[0]
    assert "AssertionError: Expected a lazy Dask array before .compute()" in shown[0]

    grading.check_code(lambda: None)
    assert "Code test passed: Module 04, Coding with AI" in shown[1] and "COMPLETE" not in shown[1]

    fake_judge(monkeypatch, True, True)
    grading.check("q", {"Eager": "e", "Why": "w"}, 2, "explanation")
    assert "MODULE 04<br>COMPLETE" in shown[2]
    assert progress.completed_modules(path=pfile) == {"04"}


def test_check_code_shows_any_exception_from_the_learners_code(monkeypatch, shown, tmp_path):
    make_module(tmp_path, monkeypatch, "04-coding-practices", "Module 04: Coding with AI")
    grading.check_code(lambda: {}["missing"])
    assert "KeyError: &#x27;missing&#x27;" in shown[0]


def test_check_code_requires_a_callable(shown):
    with pytest.raises(TypeError):
        grading.check_code("not a function")


def test_reset_progress(monkeypatch, shown, module02, capsys):
    fake_judge(monkeypatch, True, True, True, True)
    grading.check("q", POINTS, 3, "a")
    grading.reset_progress()
    assert not module02.exists()
    assert "reset" in capsys.readouterr().out


def test_show_uses_ipython_display(monkeypatch):
    shown = []
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", shown.append)
    grading._show("<b>hi</b>")
    assert shown[0].data == "<b>hi</b>"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_check.py -q`
Expected: FAIL with `AttributeError: module 'grading' has no attribute '_show'` (or `check`).

- [ ] **Step 3: Implement the public API**

Replace `grading/__init__.py` with:

```python
"""Grading for the exercise notebooks.

    from grading import check           # grade an answer, show the card, record progress
    from grading import check_code      # run a deterministic test as a card (module 04)
    from grading import reset_progress  # start your progress over
"""
from pathlib import Path

from grading import card, progress
from grading.gemini import GradingError, judge

__all__ = ["check", "check_code", "reset_progress", "GradingError"]


def _show(html: str) -> None:
    from IPython.display import HTML, display

    display(HTML(html))


def _validate(question, points, needed, answer) -> None:
    if not isinstance(points, dict) or not points:
        raise ValueError("points must be a non-empty dict of {point name: what a met point looks like}")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in points.items()):
        raise ValueError("points must map point names (str) to descriptions (str)")
    if isinstance(needed, bool) or not isinstance(needed, int) or not 1 <= needed <= len(points):
        raise ValueError(f"needed must be a whole number from 1 to {len(points)}")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("question and answer must be strings")


def _record(module: str | None, check_name: str, passed: bool):
    """(attempt, module_complete, progress view); all empty outside a module."""
    if module is None:
        return None, False, None
    result = progress.record(module, check_name, passed)
    complete = progress.completed_modules() | ({module} if result.module_complete else set())
    view = card.ProgressView(modules=progress.list_modules(), complete=complete, current=module, note=result.note)
    return result.attempt, result.module_complete, view


def check(question: str, points: dict[str, str], needed: int, answer: str) -> None:
    """Grade `answer`, show the result card, and record progress."""
    _validate(question, points, needed, answer)
    module, title = progress.module_info(Path.cwd())
    try:
        judgement = judge(question, points, answer)
    except GradingError as exc:
        _show(card.error_card(exc.kind, exc.detail))
        return
    passed = sum(p.met for p in judgement.points) >= needed
    attempt, module_complete, view = _record(module, "answer", passed)
    _show(card.verdict_card(module=module, title=title, judgement=judgement, needed=needed, passed=passed,
                            attempt=attempt, module_complete=module_complete, progress=view))


def check_code(test_fn) -> None:
    """Run `test_fn()`; returning normally passes, any exception is "not yet"."""
    if not callable(test_fn):
        raise TypeError("check_code needs a function to call, e.g. check_code(run_test)")
    module, title = progress.module_info(Path.cwd())
    try:
        test_fn()
        passed, error = True, None
    except Exception as exc:  # the learner's code failing is the point of the test
        passed, error = False, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    attempt, module_complete, view = _record(module, "code", passed)
    _show(card.code_card(module=module, title=title, passed=passed, error=error, attempt=attempt,
                         module_complete=module_complete, progress=view))


def reset_progress() -> None:
    """Delete your saved progress so every module starts over."""
    progress.reset()
    print("Your exercise progress has been reset.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_check.py -q`
Expected: all pass (17 tests).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (74 tests).

- [ ] **Step 6: Commit**

```bash
git add grading/__init__.py tests/test_check.py
git commit -m "Add check(), check_code() and reset_progress() for the notebooks"
```

---

### Task 5: Convert the template and modules 01–03 notebooks

**Files:**
- Modify: `modules/_template/exercise.ipynb`, `modules/01-how-ai-works/exercise.ipynb`, `modules/02-prompt-engineering/exercise.ipynb`, `modules/03-skills-use-and-building/exercise.ipynb`

**Interfaces:**
- Consumes: `grading.check` (Task 4).
- Produces: each notebook's import cell is the standard boilerplate ending `from grading import check`; its locked rubric cell defines `question`, `points`, `needed`; its last code cell calls `check(question, points, needed, <answer>)`.

- [ ] **Step 1: Write the notebook structure check and run it (fails)**

Save as `/tmp/check_notebooks.py` (a verification script, not committed; the spec keeps notebook execution out of CI):

```python
"""Usage: python /tmp/check_notebooks.py <module folder>... (run from the repo root)."""
import ast
import re
import sys
from pathlib import Path

import nbformat

failures = 0
for folder in map(Path, sys.argv[1:]):
    nb = nbformat.read(folder / "exercise.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    rubric = code[1]
    names = {t.id for n in ast.walk(ast.parse(rubric.source)) if isinstance(n, ast.Assign) for t in n.targets}
    problems = []
    if not re.search(r"^from grading import check(, check_code)?$", code[0].source, re.M):
        problems.append("import cell doesn't import check")
    if rubric.metadata.get("editable") is not False or rubric.metadata.get("deletable") is not False:
        problems.append("rubric cell not locked")
    if not {"question", "points", "needed"} <= names or "rubric" in names:
        problems.append(f"rubric cell defines {sorted(names)}")
    if "check(question, points, needed," not in code[-1].source:
        problems.append("last cell doesn't call check(question, points, needed, ...)")
    if "grade(" in "".join(c.source for c in code):
        problems.append("still calls grade()")
    failures += bool(problems)
    print(("FAIL " if problems else "PASS ") + str(folder) + ("" if not problems else ": " + "; ".join(problems)))
sys.exit(1 if failures else 0)
```

Run: `.venv/bin/python /tmp/check_notebooks.py modules/_template modules/01-how-ai-works modules/02-prompt-engineering modules/03-skills-use-and-building`
Expected: 4 × FAIL (import cell, rubric cell defines `question`/`rubric`, last cell, still calls `grade()`).

- [ ] **Step 2: Rewrite the cells**

Run from the repo root (keeps cell ids and the rubric cells' lock metadata):

```bash
.venv/bin/python - <<'PYEOF'
import nbformat

IMPORT = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import check",
])

RUBRICS = {
    "_template": "\n".join([
        'question = "TODO: replace with this exercise\'s question/task"',
        "points = {",
        '    "TODO point name": "TODO: what an answer that meets this point does",',
        "}",
        "needed = 1  # how many points must be met to pass",
    ]),
    "01-how-ai-works": "\n".join([
        "question = (",
        '    "The learner asked an AI assistant a narrow factual question in their "',
        '    "field, without supplying source material, then asked it to cite "',
        '    "sources. Grade their analysis below: grade the checking, not whether "',
        '    "the AI happened to be right."',
        ")",
        "points = {",
        '    "Checked a claim": (',
        '        "names at least one specific claim, number, or citation from the AI\'s "',
        '        "answer, says how they checked it against a source they trust, and "',
        '        "reports the result (confirmed, wrong, or couldn\'t be found). A correct "',
        '        "AI answer counts if the checking is real and specific; \'it looked "',
        '        "right\' is not checking"',
        "    ),",
        '    "Reasoning mode": (',
        '        "correctly names the reasoning mode the task required (an "',
        '        "inductive/abductive recall task, not deductive rule-following), "',
        '        "with a brief reason"',
        "    ),",
        '    "Why it needed checking": (',
        '        "explains in a sentence why the answer couldn\'t be trusted without "',
        '        "checking (the model predicts likely text rather than looking facts up)"',
        "    ),",
        "}",
        "needed = 3  # all three",
    ]),
    "02-prompt-engineering": "\n".join([
        "question = (",
        "    \"The learner rewrote this weak prompt: 'Explain this model output.' \"",
        '    "(context: a netCDF file of 1959 ocean data with physical tracer and "',
        '    "zooplankton values). Grade their rewrite below."',
        ")",
        "points = {",
        '    "Goal": "says what the model should actually produce",',
        '    "Context": "names the variables, units, or dataset details that matter",',
        '    "Output": "gives a format or audience",',
        '    "Boundaries": "says what must be verified, calculated, or flagged",',
        "}",
        "needed = 3  # pass with at least 3 of the 4 parts",
    ]),
    "03-skills-use-and-building": "\n".join([
        "question = (",
        '    "The learner wrote a skill file for a recurring AI-assisted task, "',
        '    "following the anatomy: When to use, Instructions, Example, Edge "',
        '    "cases. Grade the skill file text below. Headings may vary in wording."',
        ")",
        "points = {",
        '    "When to use": "states concretely when the skill applies",',
        '    "Instructions": "gives concrete, imperative instructions, not vague advice",',
        '    "Worked example": "includes a worked example with a real input and output, not placeholders",',
        '    "Edge cases": "lists edge cases or mistakes to avoid",',
        "}",
        "needed = 4  # all four",
    ]),
}

CHECKS = {
    "_template": "check(question, points, needed, answer)",
    "01-how-ai-works": "\n".join([
        "answer = (",
        '    f"Question asked: {question_you_asked}\\n\\n"',
        '    f"AI\'s answer: {ai_answer}\\n\\n"',
        '    f"My analysis: {your_analysis}"',
        ")",
        "",
        "check(question, points, needed, answer)",
    ]),
    "02-prompt-engineering": "check(question, points, needed, rewritten_prompt)",
    "03-skills-use-and-building": "check(question, points, needed, skill_file_text)",
}

for module, rubric in RUBRICS.items():
    path = f"modules/{module}/exercise.ipynb"
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    code[0].source = IMPORT
    code[1].source = rubric
    assert code[1].metadata == {"editable": False, "deletable": False}
    code[-1].source = CHECKS[module]
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print("updated", path)
PYEOF
```

- [ ] **Step 3: Run the structure check (passes)**

Run: `.venv/bin/python /tmp/check_notebooks.py modules/_template modules/01-how-ai-works modules/02-prompt-engineering modules/03-skills-use-and-building`
Expected: 4 × PASS.

- [ ] **Step 4: Run each notebook top to bottom without a key**

The last cell must show the "no API key" card, not a traceback:

```bash
for d in modules/_template modules/01-how-ai-works modules/02-prompt-engineering modules/03-skills-use-and-building; do
  (cd $d && env GOOGLE_API_KEY= ../../.venv/bin/python -c "
import sys; sys.path.insert(0, '../..')
import nbformat, grading
cards = []
grading._show = cards.append
ns = {}
for c in [c for c in nbformat.read('exercise.ipynb', as_version=4).cells if c.cell_type == 'code']:
    exec(c.source, ns)
print('$d', 'no-key card shown' if cards and 'no API key found' in cards[-1] else 'PROBLEM')
" | grep -v '^\(Sentence\|Token\)')
done
```

Expected: four lines ending `no-key card shown`. `GOOGLE_API_KEY=` is set to empty on purpose: `grading` loads `.env` at import, and `load_dotenv` never overrides a variable that already exists, so an empty value guarantees no real API call even when `.env` holds a key.

- [ ] **Step 5: Commit (outputs are stripped by the nbstripout filter)**

```bash
git add modules/_template/exercise.ipynb modules/01-how-ai-works/exercise.ipynb modules/02-prompt-engineering/exercise.ipynb modules/03-skills-use-and-building/exercise.ipynb
git commit -m "Switch the template and modules 01-03 notebooks to check() with point rubrics"
```

---

### Task 6: Convert the module 04 notebook

**Files:**
- Modify: `modules/04-coding-practices/exercise.ipynb`

**Interfaces:**
- Consumes: `grading.check`, `grading.check_code` (Task 4).
- Produces: import cell ending `from grading import check, check_code` followed by the numpy/xarray/dask imports; the test cell ends with `check_code(run_test)`; the last cell calls `check(question, points, needed, explanation)`.

- [ ] **Step 1: Run the structure check (fails)**

Run: `.venv/bin/python /tmp/check_notebooks.py modules/04-coding-practices`
Expected: FAIL (import cell, rubric cell, last cell, still calls `grade()`).

- [ ] **Step 2: Rewrite the cells**

```bash
.venv/bin/python - <<'PYEOF'
import nbformat

path = "modules/04-coding-practices/exercise.ipynb"
nb = nbformat.read(path, as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]

code[0].source = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import check, check_code",
    "",
    "import numpy as np",
    "import xarray as xr",
    "from dask.base import is_dask_collection",
])
code[1].source = "\n".join([
    "question = (",
    '    "The learner was asked to explain, in their own words, why calling "',
    '    "`.values` inside monthly_mean (before returning) was a problem for "',
    '    "a dask-backed xarray pipeline."',
    ")",
    "points = {",
    '    "Eager computation": (',
    '        "says `.values` forces eager computation, turning the lazy "',
    '        "Dask-backed array into a NumPy array immediately"',
    "    ),",
    '    "Why it hurts": (',
    '        "says why that\'s a problem: the whole (possibly huge) dataset is "',
    '        "loaded into memory at once, instead of staying chunked/lazy until "',
    '        "`.compute()` is called"',
    "    ),",
    "}",
    "needed = 2  # both",
])
assert code[1].metadata == {"editable": False, "deletable": False}

test = next(c for c in code if "def test_monthly_mean_remains_lazy" in c.source)
test.source = "\n".join([
    "import tempfile",
    "",
    "",
    "def test_monthly_mean_remains_lazy(tmp_dir):",
    "    for i in range(2):",
    "        ds = xr.Dataset(",
    '            {"temperature": (("time", "lat", "lon"), np.full((2, 3, 4), i + 1.0))},',
    "            coords={",
    '                "time": np.arange(i * 2, i * 2 + 2),',
    '                "lat": np.arange(3),',
    '                "lon": np.arange(4),',
    "            },",
    "        )",
    '        ds.to_netcdf(tmp_dir / f"temperature_{i}.nc")',
    "",
    '    result = monthly_mean(str(tmp_dir / "temperature_*.nc"))',
    '    assert is_dask_collection(result.data), "Expected a lazy Dask array before .compute()"',
    "",
    "    computed = result.compute()",
    "    assert isinstance(computed.data, np.ndarray)",
    "    np.testing.assert_allclose(computed.values, 1.5)",
    "",
    "",
    "def run_test():",
    "    with tempfile.TemporaryDirectory() as tmp:",
    "        test_monthly_mean_remains_lazy(Path(tmp))",
    "",
    "",
    "check_code(run_test)",
])
code[-1].source = "check(question, points, needed, explanation)"
nbformat.validate(nb)
nbformat.write(nb, path)
print("updated", path)
PYEOF
```

- [ ] **Step 3: Run the structure check (passes)**

Run: `.venv/bin/python /tmp/check_notebooks.py modules/04-coding-practices`
Expected: PASS.

- [ ] **Step 4: Run the notebook without a key: code test card, then the no-key card**

```bash
cd modules/04-coding-practices && env GOOGLE_API_KEY= ../../.venv/bin/python -c "
import sys; sys.path.insert(0, '../..')
import nbformat, grading, tempfile
from pathlib import Path
grading.progress.PROGRESS_FILE = Path(tempfile.mkdtemp()) / '.progress.json'
cards = []
grading._show = cards.append
ns = {}
for c in [c for c in nbformat.read('exercise.ipynb', as_version=4).cells if c.cell_type == 'code']:
    exec(c.source, ns)
print('buggy function ->', 'not-yet card' if 'Code test not passing yet' in cards[0] else 'PROBLEM')
print('explanation ->', 'no-key card' if 'no API key found' in cards[-1] else 'PROBLEM')
fixed = '''def monthly_mean(path):
    ds = xr.open_mfdataset(path, chunks={\"time\": 30, \"lat\": 100, \"lon\": 100})
    return ds[\"temperature\"].mean(dim=\"time\")'''
exec(fixed, ns); ns['check_code'](ns['run_test'])
print('fixed function ->', 'pass card' if 'Code test passed' in cards[-1] else 'PROBLEM')
"; cd ../..
```

Expected: `buggy function -> not-yet card`, `explanation -> no-key card`, `fixed function -> pass card`. Progress goes to a temporary file, so the learner's real `.progress.json` is untouched.

- [ ] **Step 5: Commit**

```bash
git add modules/04-coding-practices/exercise.ipynb
git commit -m "Switch the module 04 notebook to check_code() and check()"
```

---

### Task 7: Docs, live checks, and visual check

**Files:**
- Modify: `CONTRIBUTING.md` (section "2a. Wire up the exercise notebook")
- Modify: `docs/superpowers/specs/2026-10-16-grading-experience-design.md` (status line only)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Update CONTRIBUTING section 2a**

In `CONTRIBUTING.md`, replace these three bullets of section 2a:

```markdown
- Keep the boilerplate import cell (`sys.path.insert(...)` +
  `from grading import grade`) — every module's notebook uses it.
```

```markdown
- Write a short, specific rubric string describing what a passing answer
  must contain — this is what the LLM judge grades against, so be
  concrete about the substance you're checking for, not the wording.
  It goes in the locked `question`/`rubric` cell near the top (cell
  metadata `"editable": false`), so learners see it before they start
  and can't change it by accident.
- If your exercise has a deterministic, checkable part (e.g. code that
  either works or doesn't), check it with a plain `assert`, not the LLM
  judge — reserve `grade()` for genuinely free-form answers.
```

with:

```markdown
- Keep the boilerplate import cell (`sys.path.insert(...)` +
  `from grading import check`) — every module's notebook uses it.
```

```markdown
- In the locked rubric cell near the top (cell metadata
  `"editable": false`), define `question`, `points` and `needed`.
  `points` maps each point's name to what an answer that meets it does;
  be concrete about the substance, not the wording. `needed` is how many
  points must be met to pass. The grader judges each point and shows a
  ✅/❌ checklist; Python counts the points against `needed`, so the
  verdict follows a fixed rule.
- The last cell is `check(question, points, needed, <answer variable>)`.
  It grades, shows the result card, and records the learner's private
  progress in `.progress.json` (git-ignored). `reset_progress()` starts
  it over.
- If your exercise has a deterministic, checkable part (e.g. code that
  either works or doesn't), wrap the test in a no-argument function and
  call `check_code(run_test)`: it shows a pass/not-yet card without the
  LLM. Reserve `check()` for genuinely free-form answers. A module that
  has both needs both to pass; list it in `REQUIRED_CHECKS` in
  `grading/progress.py`.
```

- [ ] **Step 2: Live checks with a real key**

Requires `GOOGLE_API_KEY` in `.env`. Save as `/tmp/live_checks.py` and run `.venv/bin/python /tmp/live_checks.py` from the repo root:

```python
"""Good and bad sample answers through each notebook's real cells."""
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat

sys.path.insert(0, os.getcwd())  # run from the repo root
import grading
from grading import progress

REPO = os.getcwd()
FENCE = "`" * 3  # a literal triple backtick would close this plan's code block
progress.PROGRESS_FILE = Path(tempfile.mkdtemp()) / ".progress.json"  # never touch real progress
cards = []
grading._show = cards.append

CASES = [
    ("01-how-ai-works", {"question_you_asked": "When did CalCOFI begin regular survey cruises?",
      "ai_answer": "CalCOFI began regular survey cruises in 1949. Source: CalCOFI Progress Report, 1950.",
      "your_analysis": "I checked the 1949 start date against calcofi.org's history page and the 1950 report it cited: both "
                       "confirmed. The task was inductive/abductive recall with no sources given, so it generated the most "
                       "likely answer from training patterns. It was right, but it never looked anything up, so a less "
                       "common detail could be just as fluent and wrong; that's why I checked."}, "Passed"),
    ("01-how-ai-works", {"question_you_asked": "When did CalCOFI begin?", "ai_answer": "1949.",
      "your_analysis": "Looked right. It was deductive."}, "Not yet"),
    ("02-prompt-engineering", {"rewritten_prompt": "I have a netCDF file of ocean data for the year 1959. The file has physical "
      "tracer values as well as zooplankton values. Write a Jupyter notebook to analyze the values and give the yearly trend "
      "for each of the tracers. The results need to be plotted suitable for professional publication, including PDF and PNG "
      "formats. One plot should compare the trends of the three different zooplankton types."}, "Passed"),
    ("02-prompt-engineering", {"rewritten_prompt": "Explain this model output."}, "Not yet"),
    ("03-skills-use-and-building", {"skill_file_text": open(f"{REPO}/modules/03-skills-use-and-building/README.md").read()
      .split(f"### Minimal example\n{FENCE}markdown\n")[1].split(f"\n{FENCE}")[0]}, "Passed"),
    ("03-skills-use-and-building", {"skill_file_text": "# Skill: Notes\n\n## When to use\nFor notes.\n\n## Instructions\n- Be helpful."}, "Not yet"),
    ("04-coding-practices", {"explanation": "Calling .values forced Dask to compute immediately and turned the lazy array into "
      "an in-memory NumPy array, so the whole dataset was loaded into RAM at once instead of staying chunked until .compute()."}, "Passed"),
    ("04-coding-practices", {"explanation": "It was slow because the code wasn't optimized."}, "Not yet"),
]
ok = 0
for folder, answers, expected in CASES:
    os.chdir(f"{REPO}/modules/{folder}")
    code = [c.source for c in nbformat.read("exercise.ipynb", as_version=4).cells if c.cell_type == "code"]
    ns = {}
    exec(code[0], ns)
    exec(code[1], ns)
    ns.update(answers)
    exec(code[-1], ns)
    got = "Passed" if "Passed:" in cards[-1] else "Not yet" if "Not yet:" in cards[-1] else "ERROR CARD"
    ok += got == expected
    print(f"[{'ok' if got == expected else 'MISMATCH'}] {folder}: expected {expected}, got {got}")
    time.sleep(5)
print(f"{ok}/{len(CASES)} as expected")
```

Expected: `8/8 as expected`. A mismatch means a rubric's points need rewording: fix the point description in that notebook, re-run Task 5/6 Steps 3–4 for it, and re-run this step.

- [ ] **Step 3: Visual check in light and dark**

Write real cards to files and screenshot them:

```bash
.venv/bin/python - <<'PYEOF'
from grading import card
from grading.gemini import Judgement, PointResult
view = card.ProgressView(["01", "02", "03", "04"], {"01", "02"}, "02")
j = Judgement([PointResult("Goal", True, "a notebook giving the yearly trend for each tracer."),
               PointResult("Context", True, "1959 netCDF file, physical tracers and zooplankton."),
               PointResult("Output", True, "publication-quality plots as PDF and PNG."),
               PointResult("Boundaries", False, "nothing it must verify or flag yet.")],
              "say what the model must check, e.g. flag missing values and confirm units before plotting.")
cards = [
    card.verdict_card(module="02", title="Prompt Engineering", judgement=j, needed=3, passed=True, attempt=2, module_complete=True, progress=view),
    card.verdict_card(module="02", title="Prompt Engineering", judgement=j, needed=4, passed=False, attempt=1, module_complete=False,
                      progress=card.ProgressView(["01", "02", "03", "04"], {"01"}, "02")),
    card.code_card(module="04", title="Coding with AI", passed=False, error="AssertionError: Expected a lazy Dask array before .compute()",
                   attempt=1, module_complete=False, progress=view),
    card.error_card("no_key"),
]
for theme, bg in (("light", "#ffffff"), ("dark", "#1f1f1f")):
    open(f"/tmp/cards-{theme}.html", "w").write(f'<!doctype html><body style="background:{bg};padding:16px">' + "<br>".join(cards) + "</body>")
PYEOF
for t in light dark; do google-chrome --headless=new --disable-gpu --hide-scrollbars --window-size=720,1900 --virtual-time-budget=4000 --screenshot=/tmp/cards-$t.png file:///tmp/cards-$t.html 2>/dev/null; done
```

Open `/tmp/cards-light.png` and `/tmp/cards-dark.png`. Expected: the cards match the approved mock-up (green pass with stamp, amber not-yet with ❌ marks and nudge, code card with the assertion message, blue no-key card with four steps), readable on both backgrounds. Show them to the requester before the final commit.

- [ ] **Step 4: Mark the spec implemented and commit**

In `docs/superpowers/specs/2026-10-16-grading-experience-design.md`, change the status line to `Status: Approved and implemented (plan: docs/superpowers/plans/2026-10-16-grading-experience.md)`.

```bash
git add CONTRIBUTING.md docs/superpowers/specs/2026-10-16-grading-experience-design.md
git commit -m "Document check()/check_code() and the points rubric format"
```

- [ ] **Step 5: Final full run**

Run: `.venv/bin/pytest -q && git status --short`
Expected: all tests pass; no untracked `.progress*.json` files (they're ignored) and a clean tree.
