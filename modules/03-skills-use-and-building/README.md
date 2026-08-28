# Module 03: Skill Use & Building

## Objectives
By the end of this module you will be able to:
- Use the correct vocabulary for reusable AI capabilities: workflow,
  assets, skill, plugin, connector
- Choose the smallest capability that fits a task, instead of
  over-engineering a one-off prompt into a skill (or vice versa)
- Write a basic skill file that reliably steers an LLM's behavior
- Recognize a skill working inside a real tool (Claude in VS Code) so the
  abstract idea has a concrete anchor

## Why this matters in a research context
If your lab repeats the same kind of AI-assisted task — formatting
citations a certain way, running a standard QC pass on new data, reviewing
code against a house style — writing that up once as a skill means every
lab member (and every future conversation) gets consistent output, instead
of everyone re-explaining it from scratch.

## Core concept

### Vocabulary
These terms get used loosely across products, so it's worth pinning down
what each one means here:
- **Workflow** — reusable instructions and resources for doing a
  recurring task.
- **Assets** — the templates, examples, references, and scripts a
  workflow draws on.
- **Skill** — a detailed markdown file of instructions the model follows.
  This is the plain-language umbrella term used in this repo; some
  products call the same idea a "plugin."
- **Plugin** — an installable bundle of capabilities (can include one or
  more skills, plus code).
- **Connector** — live data access or controlled actions in another
  service (e.g. reading your calendar, querying a database) — this is
  about *authorized access*, not just instructions.

The distinction that matters most: **a skill tells the model how to work;
a connector gives it authorized access to information or actions.** Most
research workflows only need the former.

### Choose the smallest capability that fits
- **Prompt** — a one-off task where the context is already available in
  the conversation. No file needed.
- **Skill** — a recurring workflow with a stable method or format. Worth
  writing down once, reused many times.
- **Plugin or connector** — needed when the task requires live data or
  taking actions in another system. Keep a human reviewing anything
  consequential — this repo doesn't cover building these yet.

A quick self-check: *would I explain this the same way to a new lab member
every time?* If yes, it's at least a skill candidate. If it also needs to
reach outside the conversation (pull today's data, take an action), it's a
connector/plugin problem, not a skill problem.

### Anatomy of a good skill file
A skill is a **reusable, written-down set of instructions** — the AI
equivalent of a lab SOP, formal enough for a machine to follow
consistently. Good ones share a shape:

1. **When to use it** — trigger conditions, stated concretely (this is
   what lets an assistant self-select the right skill, and what lets a
   teammate know when to reach for it)
2. **The instructions themselves** — imperative, specific, in priority
   order. "Always X." "Never Y." "If ambiguous, default to Z."
3. **A worked example or template** — showing the input/output pattern,
   not just describing it
4. **Edge cases** — the mistakes you've actually seen happen

Real skill files can get long and detailed — a design-and-implementation
pair for modernizing a legacy scientific codebase, for instance, might run
a ten-page design document plus a hundred-plus-page implementation skill.
The value isn't length, though; it's **consistency**. A short, clear skill
that's actually followed beats a long one that isn't.

### Minimal example
```markdown
# Skill: Lab Meeting Notes Formatting

## When to use
Use when asked to turn raw/rough meeting notes into a distributable summary.

## Instructions
- Structure output as: Attendees, Decisions, Action Items (with owner),
  Open Questions.
- Action items must each have a named owner — if none is stated in the
  notes, write "OWNER NEEDED" rather than guessing.
- Keep Decisions to one line each; no editorializing.
- Do not invent action items that weren't discussed.

## Example
Input: [raw notes]
Output: [formatted summary matching the structure above]
```

### Keeping skills portable across LLMs
Because a skill is just markdown, it works anywhere you can paste text
into a system prompt, custom instructions, or a project's context — that's
exactly what this repo's `llm-context/SKILLS.md` file is: several skills
bundled into one portable reference. Skill markdown files have almost
universal compatibility across current AI tools for this reason.

---

## Appendix: seeing a skill run inside a real tool

The concepts above are tool-agnostic on purpose. This appendix grounds
them in one concrete setup — Claude running inside VS Code — so you can
see what "a skill changes the model's behavior" actually looks like
end to end. If you use a different editor or assistant, the underlying
pattern (a markdown instruction file + trigger keyword) transfers even
if the exact menus don't.

### Setup
1. Install the Claude extension from the VS Code Extensions marketplace
   (the cube icon in the sidebar → search "Claude" → Install).
2. Inside the Claude panel, add a skill by pasting a GitHub repo address
   and selecting Install. One widely used example is
   [Superpowers](https://claude.com/plugins/superpowers) — a skill
   describing good working habits for an AI assistant on real
   engineering tasks (the idea being that it's inefficient to retrain a
   general model on every team's specific working conventions, so you
   hand it the conventions as a skill instead).
3. Useful shortcuts once installed: `Ctrl-Esc` references the current
   file; highlight code + `Ctrl-Esc` references just the selected lines;
   `/keyword` invokes a specific skill (e.g. `/superpowers`).

### What it looks like in practice
A researcher had a shell script that branches a climate model run and
turns on a marine-cloud-brightening intervention when the central Pacific
warms past a threshold. They wanted to automate the monitoring step too,
and invoked the Superpowers skill rather than just asking directly:

> "I would like to automate this more, with an addition script that runs
> and checks for warming. If warming is happening, branch the run and turn
> on MCB for three months."

Because the skill's instructions call for treating this as a genuinely new
subsystem (not a small tweak), the model didn't just start generating
code. It followed the skill's process — **questions → approaches → design
→ spec** — asking a clarifying question first:

> "Right now, each time you run the .sh script to create a new branch
> case, do you hand-edit the variables at the top yourself, or is there
> already a fixed pattern the automation could compute?"

Once answered, it presented two concrete design approaches with explicit
trade-offs (a single stateful orchestrator script vs. separate scripts per
stage) and a recommendation with reasoning, before writing any code.

**Why this matters:** this back-and-forth wasn't the model being chatty —
it's the skill file doing exactly what "the instructions themselves" and
"when to use it" are for in the anatomy above. A well-written skill
changes *how* the model approaches a category of task, not just what it
says on this one occasion.

### Other skill examples worth browsing
- [Awesome GPT Prompt Engineering](https://github.com/snwfdhmp/awesome-gpt-prompt-engineering)
- [Claude Skills (community collection)](https://github.com/alirezarezvani/claude-skills)
- [AI for Grant Writing](https://github.com/eseckel/ai-for-grant-writing#prompt-engineering)
- [Claude Superpowers](https://github.com/obra/superpowers)

## Hands-on exercise
**Task:** Pick one task you or your lab does repeatedly with AI help
(data cleaning conventions, a standard email reply, a code review
checklist). Write a skill file for it using the four-part anatomy above.
Then test it: open a fresh conversation, paste the skill, and give the
model a real example of that task.

**Success looks like:** The model's output follows your stated rules
without you re-explaining them in the prompt itself — and you can hand
the same skill file to a labmate and get the same behavior.

## Key takeaways
- Vocabulary matters: a skill tells the model *how* to work; a connector
  gives it authorized *access*. Most research tasks need only the former.
- Choose the smallest capability that fits: one-off → prompt, recurring →
  skill, needs live data/actions → plugin/connector (with human review)
- A good skill file states: when to use it, imperative instructions, a
  worked example, and known edge cases
- Consistency, not length, is the point of a skill
- Because skills are just markdown, they're portable across LLMs and
  tools — the VS Code/Superpowers example is one instance of a pattern
  that works anywhere
