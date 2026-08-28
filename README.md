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

## Learning path (current modules)

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
