# Interactive Modules Design

Date: 2026-08-28
Status: Approved for planning

## Problem

`AI_for_SIO` is a self-guided primer for research group members with no
assumed AI/coding background. It has four working modules
(`01-how-ai-works`, `02-prompt-engineering`, `03-skills-use-and-building`,
`04-coding-practices`), each following the CONTRIBUTING.md template
(Objectives → Why it matters → Core concept → Hands-on exercise → Key
takeaways). Every module's "Hands-on exercise" is currently prose only —
"go ask an AI assistant X" — with no artifact produced in the repo and no
feedback loop. The learner self-assesses against a "Success looks like"
description.

The goal is to make the hands-on portion of each module genuinely
interactive: something the learner runs and gets feedback on, within a
~30 minute module budget, without assuming the learner has a coding
background or a paid LLM subscription.

## Approach

Two mechanisms, chosen after ruling out alternatives:

- **Jupyter notebooks** as the interactive surface. Rejected alternatives:
  GitHub-native mechanisms (Issues/Discussions/Actions as the interaction
  point) — the requester confirmed "GitHub interactions" wasn't a specific
  requirement, just an open question, so it's dropped in favor of a single
  consistent mechanism. Cloud notebook runtimes (Colab, Codespaces) were
  considered for lower setup friction but rejected in favor of local
  Jupyter via `requirements.txt` — no account/quota dependency, and this
  research group already has the standing assumption of a working Python
  environment (module 04 already assumes one).
- **LLM-as-judge autograding** for exercises whose correct answer is
  free-form (a rewritten prompt, a written explanation, a drafted skill
  file) rather than a fixed value an `assert` can check. Rejected
  alternatives: a paid frontier API (cost risk for a research group with no
  guaranteed LLM budget) and a fully local model via Ollama (real install
  friction — multi-GB download, hardware dependency — and weaker judgment
  quality from small local models). Landed on a **hosted free-tier API**
  (Google Gemini) — no credit card, low-friction key signup, better
  judgment quality than small local models.

Everything else in the repo (module content, the five-section README
template, `llm-context/SKILLS.md` sync process) is unchanged. This design
only adds an interactive/checkable layer on top.

## Components

### 1. Shared grading module (`grading.py`, repo root)

A single-file module, not a package — no build/install step needed.
Notebooks add the repo root to `sys.path` in a standard first cell so they
can `from grading import grade`.

```python
def grade(question: str, rubric: str, answer: str) -> dict:
    """Calls the Gemini free-tier API, judging `answer` to `question`
    against `rubric`. Returns {"pass": bool, "feedback": str}."""
```

- Uses the official Google Gemini Python SDK.
- Reads `GOOGLE_API_KEY` from the environment via `python-dotenv` loading
  a repo-root `.env` file (gitignored). A committed `.env.example` holds
  the placeholder `GOOGLE_API_KEY=`.
- If the key is missing or invalid, raises a clear, actionable error
  (setup instructions), not a raw SDK stack trace.
- Prompts the Gemini model to act as a strict-but-fair grader against the
  supplied rubric text and return structured pass/fail + feedback.
- Rubrics are plain strings, written inline in a non-editable cell near
  the top of each notebook — no separate rubric file format.

### 2. Per-module notebooks (`modules/0N-*/exercise.ipynb`)

Every module gets an `exercise.ipynb` next to its `README.md`. The
README's "Hands-on exercise" section keeps its existing Task / Success
criteria prose (source of truth, human-readable) and gains one line:
"Do this in `exercise.ipynb`." The notebook is where the learner actually
works.

Tailored by module type — conceptual modules get light notebooks that
wrap the existing external-chat exercise with a paste-your-answer +
auto-grade step; the coding module gets a full notebook with real
executable code and a real test, reserving the LLM judge for the one part
of the exercise that's genuinely free-form (matches the repo's own
deductive/abductive framing: deterministic checks stay deterministic,
LLM-judged checks are reserved for judgment calls).

| Module | Notebook content | What gets LLM-graded |
|---|---|---|
| 01 How AI Works | Small tokenization demo cell (visualize a sentence splitting into tokens) + the existing exercise (ask an AI a narrow factual question, request citations) | Learner pastes the question, the AI's answer, and their own analysis; rubric checks they identified a fabricated/unverifiable claim and correctly named the reasoning mode (deductive/inductive/abductive) |
| 02 Prompt Engineering | Learner writes their rewritten prompt (from the netCDF/1959 exercise) as a string variable | Rubric checks the rewrite covers goal, context, output, and boundaries per the module's four-part framework |
| 03 Skills Use & Building | Learner pastes their drafted skill file text | Rubric checks for the four required sections (When to use / Instructions / Example / Edge cases) and that instructions are concrete/imperative, not vague |
| 04 Coding with AI | The `monthly_mean` buggy function as a runnable cell; a real `pytest`-style assert cell for the fix (deterministic — no LLM involved) | A free-text "explain in one sentence why this was slow" cell, graded against a rubric checking for the eager-computation/`.values` explanation |

### 3. Repo infra

New root files:
- `requirements.txt` — `jupyter`/`jupyterlab`, the Gemini SDK, `python-dotenv`,
  plus `xarray`, `dask`, `numpy`, and a netCDF engine (`netCDF4` or
  `h5netcdf`) needed to actually run module 04's `monthly_mean` example
  and its in-memory test fixtures inside the notebook
- `.env.example` — `GOOGLE_API_KEY=`
- `grading.py` — the shared grader (component 1)
- `.gitignore` additions: `.env`, `.ipynb_checkpoints/`

### 4. Governance doc updates

- `README.md` — new "Setup" section: clone → `pip install -r
  requirements.txt` → get a free Gemini API key (Google AI Studio, no
  credit card) → copy `.env.example` to `.env` and fill in the key →
  launch Jupyter. Module table gains a note that modules include a
  notebook.
- `CONTRIBUTING.md` — module-creation steps gain: copy the template's
  `exercise.ipynb` too; write a short rubric string; use the standard
  `sys.path`/`grading` import boilerplate; keep the notebook itself short
  — the 30-minute budget is protected by notebook brevity, not by cutting
  README prose.
- `modules/_template/` — gains a skeleton `exercise.ipynb`: boilerplate
  import cell, placeholder answer cell, placeholder grading cell.

### Retrofit

Modules 01-04 each get their `exercise.ipynb` per the table above.
Existing README "Hands-on exercise" sections get the one-line notebook
pointer added; task/success-criteria prose is otherwise unchanged.

## Explicitly out of scope

- No CI pipeline, no `nbformat` linting, no automated notebook execution
  checks. Verification is manual: run each notebook end-to-end once with
  a real free API key before merging. This matches the repo's scale (a
  handful of self-guided modules, not a software product).
- No GitHub-native interaction mechanism (Issues/Discussions/Actions as
  part of the exercise flow) — confirmed not a specific requirement.
- No support for alternate grading providers (OpenAI, local Ollama, etc.)
  or a provider-abstraction layer. Single hard-coded call to the Gemini
  free tier in `grading.py`. If cost/quota/provider needs change later,
  that's a small, localized follow-up to one file.
- The planned `github-basics` module and any future modules are not part
  of this pass; they'll follow the same notebook + grading.py pattern
  once written.

## Testing / validation

No automated test suite. Validation is: for each of the 4 retrofitted
notebooks, run all cells top-to-bottom with a real (free-tier) API key
and confirm (a) the deterministic parts execute correctly, and (b) the
`grade()` call returns a sensible pass/fail and non-generic feedback for
both a correct and a clearly-wrong sample answer.
