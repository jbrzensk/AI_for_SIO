# Adding a New Module

This repo is designed to expand without turning into a mess. Follow this
process every time.

## 1. Copy the template

```bash
cp -r modules/_template modules/0N-your-module-name
```

Use a two-digit prefix (`04-`, `05-`...) so modules sort in learning order.
Use kebab-case for the folder name.

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
