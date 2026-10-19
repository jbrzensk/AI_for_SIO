# Grading Experience Design

Date: 2026-10-16
Status: Approved and implemented (plan: docs/superpowers/plans/2026-10-16-grading-experience.md)

## Problem

The exercise notebooks (`modules/0N-*/exercise.ipynb`, from the
interactive-modules spec of 2026-08-28) end in a grading cell that prints
two plain lines: `PASS` or `NEEDS WORK`, then one to three sentences of
feedback. When grading can't run (no key, invalid key, offline), the
learner sees a red Python traceback. The module READMEs have since been
redesigned with pictures, callouts, and collapsible depth; the notebooks
now look flat by comparison, and the grading step gives no sense of
progress or reward.

Goal: make grading as presentable as the READMEs and more fun for the
humans doing the exercises, without changing what is being taught or
how strictly it is judged.

## Decisions made with the requester

- **All four experience features:** a styled result card, a per-point
  checklist, celebration and encouragement, and progress across modules.
- **Progress is private to the learner:** stored on their own computer,
  never committed, never sent anywhere. No accounts, no server.
- **Approach A:** a single `check()` call renders an HTML card (inline
  CSS, CSS-only animation, no JavaScript, no new dependencies). Rejected:
  `ipywidgets` (new dependency; widgets can fail to render in VS Code or
  older Jupyter, leaving a blank cell, the worst beginner experience) and
  plain styled text (can't deliver a card, checklist, or celebration).
- **Card design:** as shown in the approved mock-up
  (`card-design.html`, reviewed 2026-10-16), with **both** celebrations on
  a pass: confetti and a "MODULE NN COMPLETE" stamp.

## The card

One design, identical in every module, defined once in code. It has its
own white background, so it reads the same in light (JupyterLab) and dark
(VS Code) notebook themes. All CSS is scoped under one class prefix so it
can't clash with the notebook's own styles.

| State | Header | Body | Footer |
|---|---|---|---|
| **Passed** | Green, 🎉, "Passed: Module NN, Title", "k of n points (needed m) · attempt a"; confetti plus the "MODULE NN COMPLETE" stamp | ✅/➖ line per point with a one-line note; **"Next improvement"** (one concrete step) | Progress strip |
| **Not yet** | Amber, 🌱, "Not yet: Module NN, Title", same subtitle | ✅/❌ line per point with note; **"Your next step"**; "Most people need two or three tries. Edit your answer above, run that cell, then run this one again." | Progress strip |
| **Couldn't grade** | Blue, with an icon per problem (🔑 no or invalid key, 📡 offline, ⏳ rate limit, 🔧 unreadable reply or service error), "Couldn't grade yet: <reason>", "Your answer is fine; …" | Numbered fix for that specific problem (see Errors) | none |

- **Progress strip:** one pill per module folder found under `modules/`
  (`NN-*`, so a future module 05 appears automatically): filled green with
  ✓ when complete, outlined when not, the current module highlighted, and
  "k of N complete".
- **Celebration:** confetti pieces are generated in Python with fixed
  positions and colours (deterministic, no script) and animate with CSS
  keyframes; the stamp pops in with a CSS transform. Both play once. Under
  `prefers-reduced-motion: reduce`, animations are disabled and the final
  state is shown.
- **Text:** all learner-visible strings are fixed in `card.py`, except the
  per-point notes and next step, which come from the grader.

## Architecture

`grading.py` becomes a small folder at the repo root, still importable
without installation because notebooks add the repo root to `sys.path`:

```
grading/
  __init__.py   # public API: check, check_code, reset_progress, GradingError
  gemini.py     # one Gemini call: per-point judgement + next step
  card.py       # HTML card for each state
  progress.py   # .progress.json read/update/reset
```

The import line in notebooks stays `from grading import ...`. The old
`grade(question, rubric, answer)` function is removed; every notebook
switches to `check()`. Tests are split one file per job:
`tests/test_gemini.py` (absorbing the existing `tests/test_grading.py`
cases, adapted to the per-point contract), `tests/test_card.py`,
`tests/test_progress.py`, and `tests/test_check.py`.

**Deviation from the 2026-08-28 spec:** that spec called for a single-file
module. A folder keeps its "no package, no build step" intent; it is used
because the module now has three distinct jobs (grade, render, track).

### Public API

```python
check(question: str, points: dict[str, str], needed: int, answer: str) -> None
check_code(test_fn: Callable[[], object]) -> None
reset_progress() -> None
```

- `check()` grades `answer`, displays the card (via
  `IPython.display.display(HTML(...))`), records progress for check
  `"answer"`, and returns `None` so no extra repr appears under the card.
- `check_code()` calls `test_fn()`. Returning normally is a pass; any
  exception raised by the learner's code (an `AssertionError` or
  anything else) is "not yet", with the exception type and message shown
  on the card. Records progress for check `"code"`. No API call.
- Module number and title are inferred from the notebook's working
  directory: folder `NN-slug` gives module `NN`; the title is the text after
  `Module NN: ` in that folder's `README.md` first heading, falling back to
  the slug. Outside a module folder (e.g. `_template`), the card says
  "Exercise" and no progress is recorded.
- Calling `check()` with the wrong arguments (e.g. `needed` larger than
  the number of points, empty `points`) raises an ordinary `ValueError`;
  these are author bugs and must not be hidden behind a card.

### Rubric format in notebooks

The locked rubric cell (`editable`/`deletable` false, right after the
import cell) defines `question`, `points` (name → what a met point looks
like), and `needed`. The last cell is `check(question, points, needed,
<answer variable>)`.

| Module | Points | Needed |
|---|---|---|
| 01 | Checked a specific claim against a trusted source, with the result · Named the reasoning mode the task required (inductive/abductive recall, not deductive) with a reason · Explained why the answer couldn't be trusted unchecked (it predicts likely text, it doesn't look facts up) | 3 of 3 |
| 02 | Goal · Context · Output · Boundaries | 3 of 4 |
| 03 | When to use · Concrete, imperative instructions · A worked example · Edge cases | 4 of 4 |
| 04 | Says `.values` forces eager computation into a NumPy array · Says why that hurts (the whole dataset in memory at once, instead of lazy/chunked until `.compute()`) | 2 of 2 |

Each module's point descriptions carry over the substance of its current
rubric (including the module 01 "grade the checking, not whether the AI
erred" wording and module 02's "three of four" rule).

### Grading contract (`gemini.py`)

- One `generate_content` call per `check()`, as today: model
  `gemini-3.5-flash-lite`, `temperature=0`, automatic function calling
  off, JSON response with a schema.
- Response schema: `{"points": [{"name": string, "met": boolean,
  "note": string}], "next_step": string}`, with notes one sentence each
  and `next_step` one concrete action (on a pass: the next improvement).
- The prompt lists the points by name with their descriptions and asks
  for a judgement per point; it does not ask for an overall verdict.
- **Python decides the verdict:** passed = (number of points met) ≥
  `needed`. The model never decides pass/fail as a whole, which removes
  the borderline-flip inconsistency seen with module 02's single rubric.
- The returned point names must match `points` exactly (same set);
  otherwise the reply is treated as unreadable.

## Progress (`progress.py`)

- File: `.progress.json` at the repo root, added to `.gitignore`.
- Per module: which checks have passed, the number of graded attempts,
  and the date of the first full pass.
- Module complete when all of its required checks have passed at least
  once: `{"answer"}` for every module, except module 04 which requires
  `{"code", "answer"}`.
- **Attempts** count only real grading runs (`check()` that reached a
  verdict, or `check_code()`); "couldn't grade" runs don't count.
- **Passes stick:** a later "not yet" never un-completes a check or module.
- **Unreadable file:** moved aside to `.progress.json.bak`, progress starts
  fresh, and the card shows a one-line note. Grading never fails because of
  the progress file.
- `reset_progress()` deletes the file; documented in CONTRIBUTING.

## Errors

Learner-fixable problems render the "couldn't grade" card instead of a
traceback. Each maps from a `GradingError` with a `kind`:

| kind | Trigger | Card message / fix |
|---|---|---|
| `no_key` | `GOOGLE_API_KEY` unset or empty | The 2-minute setup steps (get key, copy `.env.example` to `.env`, paste key, restart the kernel and run all cells from the top) |
| `invalid_key` | API rejects the key (400/401/403 key errors) | Check the key in `.env`, then restart the kernel and run all cells from the top |
| `offline` | `httpx` network/timeout error | Check your internet connection and run the cell again |
| `rate_limit` | API 429 / resource exhausted | Wait a minute and try again; the free daily limit resets tomorrow |
| `unreadable` | Reply isn't valid JSON or point names don't match | Run the cell again; your answer wasn't the problem |
| `service` | Any other API error | Google's service had a problem; try again in a minute |

Programming errors (wrong arguments to `check()`) raise normally.

## Notebook and docs changes

- All five `exercise.ipynb` files (modules 01–04 and `_template`): the
  locked rubric cell becomes `question` / `points` / `needed`; the
  grading cell becomes a `check(...)` call; module 04's test cell calls
  `check_code(test_monthly_mean_remains_lazy)`. The import cell imports
  `check` (and `check_code` in module 04).
- `CONTRIBUTING.md` section 2a: the points/needed format, `check()`,
  `check_code()`, and `reset_progress()`.
- `.gitignore`: `.progress.json` and `.progress.json.bak`.

## Testing

1. **Unit tests (pytest, no API key, Gemini mocked):** the pass rule
   (boundaries of `needed`); parsing per-point replies, including
   malformed JSON and mismatched point names; each card state's HTML
   (pass contains confetti and stamp and "Next improvement"; not-yet
   contains "Your next step" and the retry nudge; each error kind shows its
   specific fix); progress (created, updated, attempt counting, passes
   stick, module 04 needs both checks, unreadable file moved aside); module
   and title inference from the folder and README; `check_code` pass and
   fail; argument validation raises `ValueError`.
2. **Live checks (real key):** good and bad sample answers for every module
   through the real notebook cells, as in the previous manual checks,
   asserting both the verdict and that each point's judgement is sensible.
3. **Visual check:** cards produced by the real `check()` are rendered and
   screenshotted on light and dark backgrounds, and shown to the requester
   before committing.

## Out of scope

- Sharing progress with anyone (group dashboards, reporting).
- Interactive widgets, buttons, or hint-on-demand.
- Changing what any exercise asks or how strictly it is judged, beyond
  restating each rubric as explicit points.
- Notebook execution in CI (still out of scope, per the 2026-08-28 spec).
