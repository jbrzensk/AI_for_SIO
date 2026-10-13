# AI for Research — Self-Guided Primer

A modular, self-guided walkthrough for using AI effectively in a research
workflow. Built to grow: start with the AI fundamentals, add modules like
"GitHub basics" or "coding practices" whenever your group needs them.

## How this repo is organized

- **`modules/`** — one folder per lesson. Each follows the same template:
  objectives → core concept → hands-on exercise → key takeaways. Work
  through them in order, or jump to whichever one is relevant.
- **`llm-context/SKILLS.md`** — a single consolidated markdown file written
  *for an LLM*, not a person. Paste it into a system prompt, a project's
  custom instructions, or a Claude Skill/CustomGPT to give any model the
  condensed version of everything in this repo. See "Keeping the LLM file
  in sync" below.
- **`CONTRIBUTING.md`** — how to add a new module without breaking the
  pattern.

## Setup

Each module now includes a runnable Jupyter notebook (`exercise.ipynb`)
alongside its `README.md`. To run them:

1. Clone this repo and `cd` into it.
2. Create and activate a virtual environment, which keeps these packages
   separate from the rest of your computer's Python (many systems refuse
   a plain `pip install` without one):
   - macOS/Linux: `python3 -m venv .venv`, then `source .venv/bin/activate`
   - Windows: `py -m venv .venv`, then `.venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Get a free Gemini API key (no credit card required) from
   [Google AI Studio](https://aistudio.google.com/apikey) — this powers
   the automated feedback on exercise answers.
5. Copy `.env.example` to `.env` and paste your key in:
   `GOOGLE_API_KEY=your-key-here`
6. Launch Jupyter: `jupyter lab` (or `jupyter notebook`), then open a
   module's `exercise.ipynb`.

Next time, just `cd` into the repo, re-activate the environment (the
`activate` command from step 2), and launch Jupyter.

The exercises themselves still ask you to go use whatever AI assistant
you already have access to — the API key above is only used internally
to grade your answers, not to run the exercises.

## Learning path (current modules)

Each module's `README.md` covers the concept; open its `exercise.ipynb`
for the hands-on part with automated feedback.

| # | Module | What you'll learn |
|---|--------|--------------------|
| 01 | [How AI Works](modules/01-how-ai-works/README.md) | AI vs. ML vs. DL, how text generation actually works (tokens → embeddings → attention → prediction), deductive/inductive/abductive reasoning, what LLMs are good/bad at |
| 02 | [Prompt Engineering](modules/02-prompt-engineering/README.md) | How to write prompts that get reliable, checkable, high-quality output |
| 03 | [Skill Use & Building](modules/03-skills-use-and-building/README.md) | What "skills" are, how to use existing ones, how to write your own, seeing one run in a real tool |
| 04 | [Coding with AI](modules/04-coding-practices/README.md) | Reproducible coding prompts, testing AI-generated fixes, personal style/design guides, Python/MATLAB/Bash examples |

### Planned / open slots
- `github-basics` — version control, branches, PRs, for research collaborators
- Add your own — see `CONTRIBUTING.md`

## Who this is for

Researchers and lab members with no assumed AI or coding background.
Each module is meant to take 20–40 minutes and end with something you did,
not just read.

## Keeping the LLM file in sync

`llm-context/SKILLS.md` is hand-maintained but mirrors the module content.
When you add or meaningfully change a module, add/update its corresponding
section in `SKILLS.md` too (there's a checklist for this in
`CONTRIBUTING.md`). Treat the modules as the source of truth for humans and
`SKILLS.md` as the compressed export for machines.
