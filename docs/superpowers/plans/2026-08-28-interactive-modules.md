# Interactive Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn each module's "Hands-on exercise" from prose-only into a runnable Jupyter notebook with an LLM-judge autograder for free-form answers, without breaking the existing module template or adding paid/heavy infra.

**Architecture:** A single shared `grading.py` at repo root wraps the Google Gemini free-tier API behind one function, `grade(question, rubric, answer) -> {"pass": bool, "feedback": str}`. Every module gets an `exercise.ipynb` next to its `README.md`; deterministic parts of an exercise (e.g. module 04's code fix) are checked with plain `assert`s, and only genuinely free-form answers are routed through `grade()`. No CI, no notebook-execution pipeline — verification is manual, matching the repo's scale.

**Tech Stack:** Python 3.12, Jupyter/JupyterLab, `google-genai` (Gemini free tier), `python-dotenv`, `xarray`/`dask`/`numpy`/`netCDF4` (module 04's existing example), `tiktoken` (module 01's tokenization demo), `pytest` (for `grading.py`'s own unit tests).

**Spec:** `docs/superpowers/specs/2026-08-28-interactive-modules-design.md`

## Global Constraints

- Local Jupyter via `requirements.txt` only — no Colab/Codespaces/cloud runtime.
- `grading.py` hard-codes the Gemini free-tier API as a single call, no provider abstraction layer.
- API keys live only in a gitignored `.env` (loaded via `python-dotenv`); a committed `.env.example` holds the placeholder `GOOGLE_API_KEY=`. Never commit a real key.
- No CI pipeline, no automated notebook-execution checks. Verification before merging is manual: run each notebook end-to-end with a real free-tier key.
- Every module's README keeps its existing five required sections (Objectives, Why this matters, Core concept, Hands-on exercise, Key takeaways) unchanged in substance — the only edit is one added line in "Hands-on exercise" pointing to `exercise.ipynb`.
- Notebook and README prose must stay LLM-provider-neutral ("an AI assistant" / "an LLM") for the *lesson content* — the Gemini dependency is internal to `grading.py` only and invisible to what the learner is told to go do.
- Keep every notebook short — the 30-minute module budget is protected by notebook brevity, not by cutting README prose.

---

## File Structure

```
grading.py                                   # new — shared LLM-judge grader
conftest.py                                  # new — empty, makes repo root importable in tests
tests/test_grading.py                        # new — unit tests for grading.py (mocked API)
requirements.txt                             # new
.env.example                                 # new
.gitignore                                   # new
README.md                                    # modified — add Setup section
CONTRIBUTING.md                              # modified — document notebook step
modules/_template/exercise.ipynb             # new — skeleton notebook
modules/01-how-ai-works/exercise.ipynb       # new
modules/01-how-ai-works/README.md            # modified — one-line pointer
modules/02-prompt-engineering/exercise.ipynb # new
modules/02-prompt-engineering/README.md      # modified — one-line pointer
modules/03-skills-use-and-building/exercise.ipynb # new
modules/03-skills-use-and-building/README.md # modified — one-line pointer
modules/04-coding-practices/exercise.ipynb   # new
modules/04-coding-practices/README.md        # modified — one-line pointer
```

---

### Task 1: Repo infrastructure — dependencies, secrets handling, ignores

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable environment providing `jupyter`, `jupyterlab`, `google-genai`, `python-dotenv`, `nbformat`, `xarray`, `dask[array]`, `numpy`, `netCDF4`, `tiktoken`, `pytest` — every later task depends on these being installed.

- [x] **Step 1: Create `requirements.txt`**

```text
jupyter
jupyterlab
nbformat
google-genai
python-dotenv
xarray
dask[array]
numpy
netCDF4
tiktoken
pytest
```

- [x] **Step 2: Create `.env.example`**

```text
GOOGLE_API_KEY=
```

- [x] **Step 3: Create `.gitignore`**

```text
.env
.ipynb_checkpoints/
.venv/
__pycache__/
*.pyc
```

- [x] **Step 4: Verify the requirements file installs cleanly**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Expected: install completes with no errors. `.venv/` now holds all dependencies. Shell state does not persist between separate commands, so every subsequent Run step in this plan that needs an installed package starts with `source .venv/bin/activate` again.

- [x] **Step 5: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "Add project dependencies, env template, and gitignore for interactive modules"
```

---

### Task 2: Shared grading module (`grading.py`)

**Files:**
- Create: `conftest.py`
- Create: `grading.py`
- Create: `tests/test_grading.py`

**Interfaces:**
- Consumes: `google-genai`, `python-dotenv` (Task 1).
- Produces: `grading.grade(question: str, rubric: str, answer: str) -> dict` returning `{"pass": bool, "feedback": str}`; `grading.GradingError` exception. Every module notebook (Tasks 3, 5-8) imports `grade` from this module.

- [x] **Step 1: Create `conftest.py` at repo root**

```python
# Empty on purpose: its presence makes pytest add the repo root to
# sys.path, so tests/test_grading.py can `import grading`.
```

- [x] **Step 2: Write the failing tests**

Create `tests/test_grading.py`:

```python
import pytest

import grading


def test_grade_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(grading.GradingError, match="GOOGLE_API_KEY"):
        grading.grade("question", "rubric", "answer")


def test_grade_returns_pass_and_feedback_from_mocked_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeResponse:
        text = '{"pass": true, "feedback": "Covers goal, context, and output."}'

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    result = grading.grade("question", "rubric", "answer")

    assert result == {
        "pass": True,
        "feedback": "Covers goal, context, and output.",
    }


def test_grade_raises_grading_error_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    class FakeResponse:
        text = "not json"

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(grading, "_get_client", lambda: FakeClient())

    with pytest.raises(grading.GradingError, match="unparseable"):
        grading.grade("question", "rubric", "answer")
```

- [x] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_grading.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grading'` (the module doesn't exist yet).

- [x] **Step 4: Implement `grading.py`**

```python
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-2.0-flash"

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["pass", "feedback"],
}


class GradingError(RuntimeError):
    """Raised when the exercise can't be graded (missing key, bad response)."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GradingError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env, add a "
            "free API key from https://aistudio.google.com/apikey, and "
            "restart the notebook kernel."
        )
    return genai.Client(api_key=api_key)


def grade(question: str, rubric: str, answer: str) -> dict:
    """Grade `answer` to `question` against `rubric` using the Gemini
    free-tier API as an LLM judge. Returns {"pass": bool, "feedback": str}.
    """
    client = _get_client()
    prompt = (
        "You are grading a research-group member's exercise answer. "
        "Be strict but fair: mark pass only if the answer satisfies the "
        "rubric's substance, not just its wording.\n\n"
        f"Question:\n{question}\n\nRubric:\n{rubric}\n\n"
        f"Learner answer:\n{answer}\n\n"
        "Respond with JSON matching the schema: "
        '{"pass": boolean, "feedback": string}. '
        "feedback should be 1-3 sentences, specific and actionable."
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )
    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GradingError(
            f"Grader returned an unparseable response: {response.text!r}"
        ) from exc
    return {"pass": bool(result["pass"]), "feedback": str(result["feedback"])}
```

- [x] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_grading.py -v`
Expected: 3 passed.

- [x] **Step 6: Commit**

```bash
git add conftest.py grading.py tests/test_grading.py
git commit -m "Add shared LLM-judge grading module with unit tests"
```

---

### Task 3: Template skeleton notebook

**Files:**
- Create: `modules/_template/exercise.ipynb`

**Interfaces:**
- Consumes: `grading.grade` (Task 2).
- Produces: the boilerplate cell pattern (sys.path insert + `from grading import grade`) that every module notebook (Tasks 5-8) copies.

- [x] **Step 1: Write the notebook builder and run it**

Run:
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

intro_md = "\n".join([
    "# Module NN Exercise: [Title]",
    "",
    "Follow the instructions in `README.md`'s Hands-on exercise section.",
    "Fill in the placeholder cells below, then run the grading cell to",
    "check your answer.",
])

boilerplate_code = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import grade",
])

answer_md = "\n".join([
    "## Your answer",
    "",
    "Replace the placeholder below with your answer.",
])

answer_code = "answer = \"TODO: replace with your answer\""

check_md = "\n".join([
    "## Check your work",
])

check_code = "\n".join([
    "question = \"TODO: replace with this exercise's question/task\"",
    "rubric = \"TODO: replace with this exercise's grading rubric\"",
    "",
    "result = grade(question, rubric, answer)",
    "print(\"PASS\" if result[\"pass\"] else \"NEEDS WORK\")",
    "print(result[\"feedback\"])",
])

nb["cells"] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(boilerplate_code),
    nbf.v4.new_markdown_cell(answer_md),
    nbf.v4.new_code_cell(answer_code),
    nbf.v4.new_markdown_cell(check_md),
    nbf.v4.new_code_cell(check_code),
]

with open("modules/_template/exercise.ipynb", "w") as f:
    nbf.write(nb, f)
PYEOF
```

- [x] **Step 2: Verify the notebook is valid**

Run: `source .venv/bin/activate && python3 -c "import nbformat; nb = nbformat.read('modules/_template/exercise.ipynb', as_version=4); nbformat.validate(nb); print('valid')"`
Expected: prints `valid`.

- [x] **Step 3: Commit**

```bash
git add modules/_template/exercise.ipynb
git commit -m "Add skeleton exercise notebook to the module template"
```

---

### Task 4: Governance doc updates (README.md, CONTRIBUTING.md)

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: nothing code-level; documents the pattern established in Tasks 1-3.

- [x] **Step 1: Add a Setup section to `README.md`**

Insert a new section after "How this repo is organized" (before "Learning path"):

```markdown
## Setup

Each module now includes a runnable Jupyter notebook (`exercise.ipynb`)
alongside its `README.md`. To run them:

1. Clone this repo and `cd` into it.
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free Gemini API key (no credit card required) from
   [Google AI Studio](https://aistudio.google.com/apikey) — this powers
   the automated feedback on exercise answers.
4. Copy `.env.example` to `.env` and paste your key in:
   `GOOGLE_API_KEY=your-key-here`
5. Launch Jupyter: `jupyter lab` (or `jupyter notebook`), then open a
   module's `exercise.ipynb`.

The exercises themselves still ask you to go use whatever AI assistant
you already have access to — the API key above is only used internally
to grade your answers, not to run the exercises.
```

- [x] **Step 2: Update the module table row description in `README.md`**

Find the "Learning path (current modules)" table and its preceding line. Add one sentence directly above the table:

```markdown
Each module's `README.md` covers the concept; open its `exercise.ipynb`
for the hands-on part with automated feedback.
```

- [x] **Step 3: Update `CONTRIBUTING.md`'s module-creation steps**

In the "## 1. Copy the template" section, change the copy command's surrounding text to mention the notebook, and add a new step after "## 2. Fill in the template sections":

```markdown
## 1. Copy the template

```bash
cp -r modules/_template modules/0N-your-module-name
```

Use a two-digit prefix (`04-`, `05-`...) so modules sort in learning order.
Use kebab-case for the folder name. This copies both `README.md` and
`exercise.ipynb`.
```

And insert this new section right after the existing "## 2. Fill in the template sections" section:

```markdown
## 2a. Wire up the exercise notebook

Edit the copied `exercise.ipynb`:
- Keep the boilerplate import cell (`sys.path.insert(...)` +
  `from grading import grade`) — every module's notebook uses it.
- Replace the placeholder `answer` cell(s) with whatever the learner
  needs to fill in for your exercise.
- Write a short, specific rubric string describing what a passing answer
  must contain — this is what the LLM judge grades against, so be
  concrete about the substance you're checking for, not the wording.
- If your exercise has a deterministic, checkable part (e.g. code that
  either works or doesn't), check it with a plain `assert`, not the LLM
  judge — reserve `grade()` for genuinely free-form answers.
- Keep the notebook short — a couple of code cells plus the grading
  cell. The 30-minute budget is protected by brevity here, not by
  cutting the README.
- In the README's "Hands-on exercise" section, add one line pointing to
  the notebook: `Do this in [exercise.ipynb](exercise.ipynb).`
```

- [x] **Step 4: Commit**

```bash
git add README.md CONTRIBUTING.md
git commit -m "Document the exercise-notebook pattern in README and CONTRIBUTING"
```

---

### Task 5: Module 01 notebook (How AI Works)

**Files:**
- Create: `modules/01-how-ai-works/exercise.ipynb`
- Modify: `modules/01-how-ai-works/README.md`

**Interfaces:**
- Consumes: `grading.grade` (Task 2), `tiktoken` (Task 1).

- [x] **Step 1: Write the notebook builder and run it**

Run:
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

intro_md = "\n".join([
    "# Module 01 Exercise: How AI Works",
    "",
    "This notebook has two parts:",
    "1. A quick demo of tokenization, so \"tokens\" stops being an abstract word.",
    "2. The hands-on exercise from the README, with automated feedback on your analysis.",
])

boilerplate_code = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import grade",
])

tokenize_md = "\n".join([
    "## Part 1: See tokenization happen",
    "",
    "Run the cell below on the sample sentence, then change `sentence` to",
    "something of your own and re-run. (First run downloads a small",
    "tokenizer file — needs internet once.)",
])

tokenize_code = "\n".join([
    "import tiktoken",
    "",
    "encoding = tiktoken.get_encoding(\"cl100k_base\")",
    "sentence = \"What color is the sky?\"",
    "token_ids = encoding.encode(sentence)",
    "tokens = [encoding.decode([t]) for t in token_ids]",
    "",
    "print(f\"Sentence: {sentence!r}\")",
    "print(f\"Token count: {len(tokens)}\")",
    "print(f\"Tokens: {tokens}\")",
    "print(f\"Token IDs: {token_ids}\")",
])

exercise_md = "\n".join([
    "## Part 2: The exercise",
    "",
    "Follow the README's instructions: ask an AI assistant a narrow",
    "factual question in your field (without giving it source material),",
    "fact-check the answer yourself, then ask it to cite sources. Paste",
    "your results below.",
])

exercise_code = "\n".join([
    "question_you_asked = \"TODO: the factual question you asked\"",
    "ai_answer = \"TODO: paste the AI's answer here\"",
    "your_analysis = (",
    "    \"TODO: which specific claim or citation is fabricated/wrong, \"",
    "    \"and which reasoning mode (deductive/inductive/abductive) this task was\"",
    ")",
])

check_md = "## Check your work"

check_code = "\n".join([
    "question = (",
    "    \"The learner asked an AI assistant a narrow factual question in their \"",
    "    \"field, without supplying source material, then asked it to cite \"",
    "    \"sources. Their analysis should be graded below.\"",
    ")",
    "rubric = (",
    "    \"Pass only if the learner's analysis (a) points to a specific claim, \"",
    "    \"number, or citation from the AI's answer that is fabricated, \"",
    "    \"unverifiable, or subtly wrong, AND (b) correctly names the reasoning \"",
    "    \"mode the task required (this is an abductive/inductive recall task, \"",
    "    \"not deductive rule-following), with a brief reason why.\"",
    ")",
    "answer = (",
    "    f\"Question asked: {question_you_asked}\\n\\n\"",
    "    f\"AI's answer: {ai_answer}\\n\\n\"",
    "    f\"My analysis: {your_analysis}\"",
    ")",
    "",
    "result = grade(question, rubric, answer)",
    "print(\"PASS\" if result[\"pass\"] else \"NEEDS WORK\")",
    "print(result[\"feedback\"])",
])

nb["cells"] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(boilerplate_code),
    nbf.v4.new_markdown_cell(tokenize_md),
    nbf.v4.new_code_cell(tokenize_code),
    nbf.v4.new_markdown_cell(exercise_md),
    nbf.v4.new_code_cell(exercise_code),
    nbf.v4.new_markdown_cell(check_md),
    nbf.v4.new_code_cell(check_code),
]

with open("modules/01-how-ai-works/exercise.ipynb", "w") as f:
    nbf.write(nb, f)
PYEOF
```

- [x] **Step 2: Verify the notebook is valid and the deterministic cell runs**

Run:
```bash
source .venv/bin/activate
python3 -c "import nbformat; nb = nbformat.read('modules/01-how-ai-works/exercise.ipynb', as_version=4); nbformat.validate(nb); print('valid')"
python3 -c "
import tiktoken
encoding = tiktoken.get_encoding('cl100k_base')
print(encoding.encode('What color is the sky?'))
"
```
Expected: `valid`, followed by a list of integer token IDs (confirms `tiktoken` and the tokenization demo logic work).

- [x] **Step 3: Add the notebook pointer to the README**

In `modules/01-how-ai-works/README.md`, in the "## Hands-on exercise" section, add this line directly under the `**Task:**` line:

```markdown
Do this in [exercise.ipynb](exercise.ipynb).
```

- [x] **Step 4: Manual end-to-end check (requires a real API key)**

If a `GOOGLE_API_KEY` is available in this environment: open the notebook, fill in `question_you_asked`/`ai_answer`/`your_analysis` with a real worked example (once with a genuinely correct analysis, once with an obviously wrong one), run all cells, and confirm `grade()` returns `PASS`/`NEEDS WORK` sensibly for each. If no key is available, skip execution and note in your task summary that this manual check is still needed from the user before merging.

- [x] **Step 5: Commit**

```bash
git add modules/01-how-ai-works/exercise.ipynb modules/01-how-ai-works/README.md
git commit -m "Add interactive exercise notebook to module 01"
```

---

### Task 6: Module 02 notebook (Prompt Engineering)

**Files:**
- Create: `modules/02-prompt-engineering/exercise.ipynb`
- Modify: `modules/02-prompt-engineering/README.md`

**Interfaces:**
- Consumes: `grading.grade` (Task 2).

- [x] **Step 1: Write the notebook builder and run it**

Run:
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

intro_md = "\n".join([
    "# Module 02 Exercise: Prompt Engineering",
    "",
    "Rewrite the weak prompt from the README (\"Explain this model",
    "output.\") into a strong one, then check it against the four-part",
    "framework (goal, context, output, boundaries).",
])

boilerplate_code = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import grade",
])

answer_md = "## Your rewritten prompt"

answer_code = "rewritten_prompt = \"TODO: your rewritten prompt goes here\""

check_md = "## Check your work"

check_code = "\n".join([
    "question = (",
    "    \"The learner rewrote this weak prompt: 'Explain this model output.' \"",
    "    \"(context: a netCDF file of 1959 ocean data with physical tracer and \"",
    "    \"zooplankton values). Grade their rewrite below.\"",
    ")",
    "rubric = (",
    "    \"Pass only if the rewritten prompt clearly specifies: (1) a goal - \"",
    "    \"what the model should actually produce; (2) context - the variables, \"",
    "    \"units, or dataset details that matter; (3) output - format/audience; \"",
    "    \"and (4) boundaries - what must be verified, calculated, or flagged. \"",
    "    \"A prompt missing two or more of these should not pass.\"",
    ")",
    "",
    "result = grade(question, rubric, rewritten_prompt)",
    "print(\"PASS\" if result[\"pass\"] else \"NEEDS WORK\")",
    "print(result[\"feedback\"])",
])

nb["cells"] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(boilerplate_code),
    nbf.v4.new_markdown_cell(answer_md),
    nbf.v4.new_code_cell(answer_code),
    nbf.v4.new_markdown_cell(check_md),
    nbf.v4.new_code_cell(check_code),
]

with open("modules/02-prompt-engineering/exercise.ipynb", "w") as f:
    nbf.write(nb, f)
PYEOF
```

- [x] **Step 2: Verify the notebook is valid**

Run: `source .venv/bin/activate && python3 -c "import nbformat; nb = nbformat.read('modules/02-prompt-engineering/exercise.ipynb', as_version=4); nbformat.validate(nb); print('valid')"`
Expected: prints `valid`.

- [x] **Step 3: Add the notebook pointer to the README**

In `modules/02-prompt-engineering/README.md`, in the "## Hands-on exercise" section, add directly under the `**Task — repair the prompt:**` line:

```markdown
Do this in [exercise.ipynb](exercise.ipynb).
```

- [x] **Step 4: Manual end-to-end check (requires a real API key)**

If a `GOOGLE_API_KEY` is available: fill in `rewritten_prompt` once with the README's own "strong version" example (expect `PASS`) and once with the README's "weak" example, `"Explain this model output."` (expect `NEEDS WORK`), running the grading cell each time. If no key is available, skip and note the pending manual check.

- [x] **Step 5: Commit**

```bash
git add modules/02-prompt-engineering/exercise.ipynb modules/02-prompt-engineering/README.md
git commit -m "Add interactive exercise notebook to module 02"
```

---

### Task 7: Module 03 notebook (Skills Use & Building)

**Files:**
- Create: `modules/03-skills-use-and-building/exercise.ipynb`
- Modify: `modules/03-skills-use-and-building/README.md`

**Interfaces:**
- Consumes: `grading.grade` (Task 2).

- [x] **Step 1: Write the notebook builder and run it**

Run:
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

intro_md = "\n".join([
    "# Module 03 Exercise: Skill Use & Building",
    "",
    "Write a skill file (per the README's four-part anatomy) for a task",
    "you or your lab does repeatedly with AI help. Paste it below to",
    "check it.",
])

boilerplate_code = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import grade",
])

answer_md = "## Your skill file"

answer_code = "skill_file_text = \"TODO: paste your full skill file markdown here\""

check_md = "## Check your work"

check_code = "\n".join([
    "question = (",
    "    \"The learner wrote a skill file for a recurring AI-assisted task, \"",
    "    \"following the anatomy: When to use, Instructions, Example, Edge \"",
    "    \"cases. Grade the skill file text below.\"",
    ")",
    "rubric = (",
    "    \"Pass only if the skill file has all four sections (When to use, \"",
    "    \"Instructions, Example, Edge cases - headings may vary in wording but \"",
    "    \"all four concepts must be present), the instructions are concrete \"",
    "    \"and imperative (not vague advice), and it includes a worked example.\"",
    ")",
    "",
    "result = grade(question, rubric, skill_file_text)",
    "print(\"PASS\" if result[\"pass\"] else \"NEEDS WORK\")",
    "print(result[\"feedback\"])",
])

nb["cells"] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(boilerplate_code),
    nbf.v4.new_markdown_cell(answer_md),
    nbf.v4.new_code_cell(answer_code),
    nbf.v4.new_markdown_cell(check_md),
    nbf.v4.new_code_cell(check_code),
]

with open("modules/03-skills-use-and-building/exercise.ipynb", "w") as f:
    nbf.write(nb, f)
PYEOF
```

- [x] **Step 2: Verify the notebook is valid**

Run: `source .venv/bin/activate && python3 -c "import nbformat; nb = nbformat.read('modules/03-skills-use-and-building/exercise.ipynb', as_version=4); nbformat.validate(nb); print('valid')"`
Expected: prints `valid`.

- [x] **Step 3: Add the notebook pointer to the README**

In `modules/03-skills-use-and-building/README.md`, in the "## Hands-on exercise" section, add directly under the `**Task:**` line:

```markdown
Do this in [exercise.ipynb](exercise.ipynb).
```

- [x] **Step 4: Manual end-to-end check (requires a real API key)**

If a `GOOGLE_API_KEY` is available: paste the README's own "Minimal example" skill file (the Lab Meeting Notes Formatting one) into `skill_file_text` — expect `PASS`. Then try a skill file text with a section missing (e.g. no Edge cases) — expect `NEEDS WORK`. If no key is available, skip and note the pending manual check.

- [x] **Step 5: Commit**

```bash
git add modules/03-skills-use-and-building/exercise.ipynb modules/03-skills-use-and-building/README.md
git commit -m "Add interactive exercise notebook to module 03"
```

---

### Task 8: Module 04 notebook (Coding with AI)

**Files:**
- Create: `modules/04-coding-practices/exercise.ipynb`
- Modify: `modules/04-coding-practices/README.md`

**Interfaces:**
- Consumes: `grading.grade` (Task 2); `xarray`, `dask`, `numpy`, `netCDF4` (Task 1).

- [x] **Step 1: Write the notebook builder and run it**

Run:
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

intro_md = "\n".join([
    "# Module 04 Exercise: Coding with AI",
    "",
    "Fix the `monthly_mean` function below so it stays lazy (Dask-backed)",
    "instead of forcing the whole array into memory, verify the fix with",
    "a real test, then explain the bug in your own words.",
])

boilerplate_code = "\n".join([
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parent.parent))",
    "from grading import grade",
    "",
    "import numpy as np",
    "import xarray as xr",
    "from dask.base import is_dask_collection",
])

buggy_md = "\n".join([
    "## The buggy function",
    "",
    "Here's the function from the README. Fix the marked line so the",
    "return value stays a lazy Dask array (hint: you'll also need to",
    "change the `.mean(...)` call, since a lazy `xarray.DataArray` uses",
    "`dim=`, not `axis=`).",
])

buggy_code = "\n".join([
    "def monthly_mean(path):",
    "    ds = xr.open_mfdataset(",
    "        path,",
    "        chunks={\"time\": 30, \"lat\": 100, \"lon\": 100},",
    "    )",
    "    temperature = ds[\"temperature\"].values  # TODO: fix this line",
    "    return temperature.mean(axis=0)",
])

test_md = "\n".join([
    "## Test your fix",
    "",
    "Run the cell below. It builds two small sample netCDF files, calls",
    "your `monthly_mean`, and checks that the result is still a lazy",
    "Dask array before `.compute()` is called.",
])

test_code = "\n".join([
    "import tempfile",
    "",
    "",
    "def test_monthly_mean_remains_lazy(tmp_dir):",
    "    for i in range(2):",
    "        ds = xr.Dataset(",
    "            {\"temperature\": ((\"time\", \"lat\", \"lon\"), np.full((2, 3, 4), i + 1.0))},",
    "            coords={",
    "                \"time\": np.arange(i * 2, i * 2 + 2),",
    "                \"lat\": np.arange(3),",
    "                \"lon\": np.arange(4),",
    "            },",
    "        )",
    "        ds.to_netcdf(tmp_dir / f\"temperature_{i}.nc\")",
    "",
    "    result = monthly_mean(str(tmp_dir / \"temperature_*.nc\"))",
    "    assert is_dask_collection(result.data), \"Expected a lazy Dask array before .compute()\"",
    "",
    "    computed = result.compute()",
    "    assert isinstance(computed.data, np.ndarray)",
    "    np.testing.assert_allclose(computed.values, 1.5)",
    "",
    "    print(\"PASS: monthly_mean stayed lazy until .compute()\")",
    "",
    "",
    "with tempfile.TemporaryDirectory() as tmp:",
    "    test_monthly_mean_remains_lazy(Path(tmp))",
])

explain_md = "\n".join([
    "## Explain the bug",
    "",
    "In 1-2 sentences, explain why the original `.values` line was a",
    "problem.",
])

explain_code = "explanation = \"TODO: your explanation here\""

check_md = "## Check your explanation"

check_code = "\n".join([
    "question = (",
    "    \"The learner was asked to explain, in their own words, why calling \"",
    "    \"`.values` inside monthly_mean (before returning) was a problem for \"",
    "    \"a dask-backed xarray pipeline.\"",
    ")",
    "rubric = (",
    "    \"Pass only if the explanation mentions that `.values` forces eager \"",
    "    \"computation / converts the lazy Dask-backed array to a NumPy array \"",
    "    \"immediately, which can load the entire (possibly huge) dataset into \"",
    "    \"memory at once, rather than keeping it chunked/lazy until `.compute()` \"",
    "    \"is explicitly called.\"",
    ")",
    "",
    "result = grade(question, rubric, explanation)",
    "print(\"PASS\" if result[\"pass\"] else \"NEEDS WORK\")",
    "print(result[\"feedback\"])",
])

nb["cells"] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(boilerplate_code),
    nbf.v4.new_markdown_cell(buggy_md),
    nbf.v4.new_code_cell(buggy_code),
    nbf.v4.new_markdown_cell(test_md),
    nbf.v4.new_code_cell(test_code),
    nbf.v4.new_markdown_cell(explain_md),
    nbf.v4.new_code_cell(explain_code),
    nbf.v4.new_markdown_cell(check_md),
    nbf.v4.new_code_cell(check_code),
]

with open("modules/04-coding-practices/exercise.ipynb", "w") as f:
    nbf.write(nb, f)
PYEOF
```

- [x] **Step 2: Verify the notebook is valid**

Run: `source .venv/bin/activate && python3 -c "import nbformat; nb = nbformat.read('modules/04-coding-practices/exercise.ipynb', as_version=4); nbformat.validate(nb); print('valid')"`
Expected: prints `valid`.

- [x] **Step 3: Verify the deterministic test cell actually fails on the buggy code and passes on the fix**

Run this against the buggy version (should fail):
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from dask.base import is_dask_collection


def monthly_mean(path):
    ds = xr.open_mfdataset(path, chunks={"time": 30, "lat": 100, "lon": 100})
    temperature = ds["temperature"].values
    return temperature.mean(axis=0)


def test_monthly_mean_remains_lazy(tmp_dir):
    for i in range(2):
        ds = xr.Dataset(
            {"temperature": (("time", "lat", "lon"), np.full((2, 3, 4), i + 1.0))},
            coords={
                "time": np.arange(i * 2, i * 2 + 2),
                "lat": np.arange(3),
                "lon": np.arange(4),
            },
        )
        ds.to_netcdf(tmp_dir / f"temperature_{i}.nc")

    result = monthly_mean(str(tmp_dir / "temperature_*.nc"))
    assert is_dask_collection(result.data), "Expected a lazy Dask array before .compute()"
    print("unexpectedly passed")


with tempfile.TemporaryDirectory() as tmp:
    try:
        test_monthly_mean_remains_lazy(Path(tmp))
    except AssertionError as e:
        print(f"failed as expected: {e}")
PYEOF
```
Expected: `failed as expected: Expected a lazy Dask array before .compute()`

Run this against the fixed version (should pass):
```bash
source .venv/bin/activate && python3 <<'PYEOF'
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from dask.base import is_dask_collection


def monthly_mean(path):
    ds = xr.open_mfdataset(path, chunks={"time": 30, "lat": 100, "lon": 100})
    return ds["temperature"].mean(dim="time")


def test_monthly_mean_remains_lazy(tmp_dir):
    for i in range(2):
        ds = xr.Dataset(
            {"temperature": (("time", "lat", "lon"), np.full((2, 3, 4), i + 1.0))},
            coords={
                "time": np.arange(i * 2, i * 2 + 2),
                "lat": np.arange(3),
                "lon": np.arange(4),
            },
        )
        ds.to_netcdf(tmp_dir / f"temperature_{i}.nc")

    result = monthly_mean(str(tmp_dir / "temperature_*.nc"))
    assert is_dask_collection(result.data)
    computed = result.compute()
    assert isinstance(computed.data, np.ndarray)
    np.testing.assert_allclose(computed.values, 1.5)
    print("PASS: monthly_mean stayed lazy until .compute()")


with tempfile.TemporaryDirectory() as tmp:
    test_monthly_mean_remains_lazy(Path(tmp))
PYEOF
```
Expected: `PASS: monthly_mean stayed lazy until .compute()`

- [x] **Step 4: Add the notebook pointer to the README**

In `modules/04-coding-practices/README.md`, in the "## Hands-on exercise" section, add directly under the `**Task:**` line:

```markdown
Do this in [exercise.ipynb](exercise.ipynb).
```

- [x] **Step 5: Manual end-to-end check (requires a real API key)**

If a `GOOGLE_API_KEY` is available: fill in `explanation` once with a correct explanation mentioning eager computation/memory (expect `PASS`) and once with an unrelated or vague explanation (expect `NEEDS WORK`). If no key is available, skip and note the pending manual check.

- [x] **Step 6: Commit**

```bash
git add modules/04-coding-practices/exercise.ipynb modules/04-coding-practices/README.md
git commit -m "Add interactive exercise notebook to module 04"
```

---

## Execution notes (2026-09-30)

Executed inline on branch `feature/interactive-notebooks`. All steps are done, including the four manual end-to-end checks (Tasks 5–8), run against the live Gemini API: 9 of 9 good/bad sample answers graded as expected. Deviations from the plan text above:

- **Task 2, model:** `MODEL_NAME = "gemini-3.5-flash-lite"`, not `gemini-2.0-flash`. Gemini 2.0 Flash was shut down on 2026-06-01, and the 2.5 models are closed to new accounts. Changing models later is a one-line edit.
- **Task 2, invalid key:** the spec requires a clear error for a missing *or invalid* key. `grade()` now wraps Gemini API errors in `GradingError` with setup guidance, covered by a fourth test (`test_grade_raises_clear_error_when_api_rejects_key`).
- **Task 1:** appended to the existing `.gitignore` (which holds `.claude/worktrees/`) instead of overwriting it.
- **Tasks 5–8:** each README pointer is its own paragraph after the Task paragraph, not inserted mid-paragraph.
- **Final review, fix 1:** per the spec, each notebook's `question`/`rubric` lives in a locked cell (`editable`/`deletable` false) right after the import cell, instead of an editable cell at the bottom. CONTRIBUTING describes this layout.
- **Final review, fix 2:** README Setup creates and activates a `.venv` before `pip install`, because a bare `pip install` fails on Ubuntu 24.04, Debian 12 and Homebrew Python (PEP 668).
- **Manual check, module 03:** the plan expected the README's "Minimal example" skill to PASS, but it had no Edge cases section and only placeholder input/output, so the grader correctly marked it NEEDS WORK. The example now has a worked example and edge cases, and grades PASS.
- **Ported from the earlier, unmerged `worktree-interactive-modules` attempt (now deleted):** network failures and replies missing a field raise `GradingError`; the module 03/04 README pointers say what the notebook covers; module 01's placeholder asks why the reasoning mode applies. Automatic function calling is also turned off, which stops the SDK printing a confusing warning into notebooks.
- **Consistency:** `grade()` runs at `temperature=0`, and the module 02 rubric no longer contradicts itself. It used to say both "pass only if all four parts" and "fail if two or more are missing", so the README's strong prompt, which lacks explicit boundaries, flipped between PASS and NEEDS WORK (4 of 5 PASS, even at temperature 0). It now says: pass with three of four parts, and name the missing one. 5 of 5 PASS afterwards, while prompts missing two parts still fail.
- **Module 01 rubric (2026-10-06):** grades the learner's fact-checking rather than whether the AI erred. The old rubric and README success criterion required finding a fabricated claim, so a correct AI answer could never pass. The README, notebook placeholder and rubric now ask for a specific claim checked against a trusted source (with the result), the reasoning mode, and why the answer couldn't be trusted unchecked.
