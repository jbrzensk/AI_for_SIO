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
  `from grading import check`) — every module's notebook uses it.
- Replace the placeholder `answer` cell(s) with whatever the learner
  needs to fill in for your exercise.
- In the locked rubric cell near the top (cell metadata
  `"editable": false`), define `question`, `points` and `needed`.
  `points` maps each point's name to what an answer that meets it does;
  be concrete about the substance, not the wording. `needed` is how many
  points must be met to pass. The grader judges each point and shows a
  ✅/❌ checklist (➖ for points a passing answer didn't need); Python
  counts the points against `needed`, so the verdict follows a fixed rule.
- The last cell is `check(question, points, needed, <answer variable>)`.
  It grades, shows the result card, and records the learner's private
  progress in `.progress.json` (git-ignored). `reset_progress()` starts
  it over.
- If your exercise has a deterministic, checkable part (e.g. code that
  either works or doesn't), wrap the test in a no-argument function and
  call `check_code(run_test)`: it shows a pass/not-yet card without the
  LLM. Reserve `check()` for genuinely free-form answers. A module that
  has both needs both to pass; list it in `REQUIRED_CHECKS` in
  `grading/progress.py`. A module with only a code test (no `check()`
  call at all) must still set its `REQUIRED_CHECKS` entry, to `{"code"}`
  — otherwise it defaults to requiring an "answer" check that never
  happens, and the module can never complete.
- Keep the notebook short — a couple of code cells plus the grading
  cell. The 30-minute budget is protected by brevity here, not by
  cutting the README.
- In the README's "Hands-on exercise" section, add one line pointing to
  the notebook: `Do this in [exercise.ipynb](exercise.ipynb).`
- Commit notebooks without outputs. `.gitattributes` runs every
  `*.ipynb` through `nbstripout` on commit, once you've run
  `nbstripout --install --attributes .gitattributes` in your clone
  (README Setup, step 3). Keep the `--attributes .gitattributes` part: a
  bare `nbstripout --install` also writes `.git/info/attributes`, which
  filters notebooks on every commit, including old ones from before
  notebooks were stripped, so those check out looking modified. To also
  drop the editor details some tools add (VS Code's `language_info`
  extras), run this once as well:
  `git config filter.nbstripout.extrakeys "metadata.language_info.codemirror_mode metadata.language_info.file_extension metadata.language_info.mimetype metadata.language_info.nbconvert_exporter metadata.language_info.pygments_lexer metadata.language_info.version"`
- If Git says notebooks changed when you didn't touch them (often right
  after a pull or checkout), someone committed them without the filter,
  for example from GitHub's web editor. Normalize them once and commit:
  `git add --renormalize -- '*.ipynb'`, then `git commit`. To get past it
  for a single command instead (e.g. a pull that refuses to run), prefix
  it with `git -c filter.nbstripout.clean=cat`. Also re-run the install
  command above if you delete or move the repo's `.venv`: the filter
  points at that exact Python, and Git refuses notebook operations while
  it's missing.

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
