# Adding a New Module

This repo is designed to expand without turning into a mess. Follow this
process every time.

## 1. Copy the template

```bash
cp -r modules/_template modules/0N-your-module-name
```

Use a two-digit prefix (`04-`, `05-`...) so modules sort in learning order.
Use kebab-case for the folder name. This copies both `README.md` and
`exercise.ipynb`.

## 2. Fill in the template sections

Every module's `README.md` must keep these five sections, in this order:

1. **Objectives** — 3-5 bullet points, "By the end you will be able to..."
2. **Why this matters in a research context** — 1-2 sentences grounding it
3. **Core concept** — the actual teaching content. Prefer short sections
   with concrete examples over long theory. Use real research-adjacent
   examples where possible (analyzing data, drafting a methods section,
   cleaning a dataset), not generic ones.
4. **Hands-on exercise** — something the learner *does*, with a clear
   success criterion. No exercise = the module isn't done.
5. **Key takeaways** — 3-6 short bullet points. These should be
   self-contained enough to lift directly into `llm-context/SKILLS.md`.

## 2a. Wire up the exercise notebook

Edit the copied `exercise.ipynb`:

- Keep the boilerplate import cell (`sys.path.insert(...)` +
  `from grading import grade`) — every module's notebook uses it.
- Replace the placeholder `answer` cell(s) with whatever the learner
  needs to fill in for your exercise.
- Write a short, specific rubric string describing what a passing answer
  must contain — this is what the LLM judge grades against, so be
  concrete about the substance you're checking for, not the wording.
  It goes in the locked `question`/`rubric` cell near the top (cell
  metadata `"editable": false`), so learners see it before they start
  and can't change it by accident.
- If your exercise has a deterministic, checkable part (e.g. code that
  either works or doesn't), check it with a plain `assert`, not the LLM
  judge — reserve `grade()` for genuinely free-form answers.
- Keep the notebook short — a couple of code cells plus the grading
  cell. The 30-minute budget is protected by brevity here, not by
  cutting the README.
- In the README's "Hands-on exercise" section, add one line pointing to
  the notebook: `Do this in [exercise.ipynb](exercise.ipynb).`

## 3. Add it to the learning path

Add a row to the table in the root `README.md`.

## 4. Update the LLM-facing file

Open `llm-context/SKILLS.md` and add a new `##` section for your module,
using the **Key takeaways** from step 2 as your starting point, rewritten
as direct instructions/facts rather than "you will learn" language. Keep
each module's section under ~150 words in that file — it's a compressed
reference, not a copy of the lesson.

## Style notes

- Write for someone with zero prior AI/coding background unless the module
  explicitly says otherwise.
- Prefer one good worked example over three shallow ones.
- Keep each module to a 20-40 minute read/do.
- Don't assume a specific LLM provider unless the module is explicitly
  about one; write "an AI assistant" / "an LLM" rather than a brand name.
