# Module 02: Prompt Engineering

## Objectives
By the end of this module you will be able to:
- Write prompts that specify goal, context, output, and boundaries
- Explain why LLMs need full sentences and commands, not search-engine
  keywords or questions
- Apply a five-step checklist (goal, evidence, output, check, confirm) to
  a real research prompt
- Take a vague prompt and rewrite it into one that produces a checkable,
  useful result

## Why this matters in a research context
The gap between "meh" and "genuinely useful" AI output is almost always
the prompt, not the model. A vague ask gets a generic answer; a specific
ask gets a specific one — and a prompt built for verification gets you an
answer you can actually trust. A prompt is a specification for the work,
not a magic phrase.

## Core concept

### A useful prompt has four parts
1. **Goal** — what should the model actually do?
2. **Context** — what evidence, files, code, or assumptions matter?
3. **Output** — what form, length, and audience?
4. **Boundaries** — what must stay fixed, be verified, or remain a draft?

Not every prompt needs all four — add whichever pieces would materially
change the answer. A good habit: before sending a prompt, ask yourself
which of the four you left out, and whether that was on purpose.

### Good → Better → Best
**Scenario:** you want an analysis of a paper you don't quite understand.

- **Good:** "Summarize this paper."
- **Better:** "Summarize the question, methods, data, and findings for an
  ocean scientist."
- **Best:** "Use only the attached paper. Separate the authors' claims from
  your interpretation, list three limitations, and cite page or section
  locations."

Each version adds one of the four parts above. "Better" adds context and
output (audience). "Best" adds boundaries — and, notably, makes the result
*easier to verify*, not just more detailed. "Best" is task-dependent: for
a quick gut-check, "Good" may be all you need.

### Google habits will work against you here
| Google (search engine) | LLM |
|---|---|
| Keywords | Full sentences |
| Minimum words required | Grammar improves the answer |
| You decide the best result from a list | Probability dictates the single response you get |
| Needs correct spelling | Spelling mistakes mostly still work |

The mental shift: you're not retrieving a document, you're specifying a
task. Write it the way you'd write it to a capable person who can't ask
you follow-up questions mid-task.

### A five-part checklist for real prompts
For anything beyond a quick question, work through:
1. **Goal** — what's the result, and who's it for?
2. **Evidence** — what files or sources actually matter?
3. **Output** — format and any real constraints?
4. **Check** — what numbers, units, or citations need verifying after?
5. **Confirm** — what assumptions, code, or scientific logic need your
   sign-off?

**Weak:** "Can you write me code to find the maximum temperature of a
netCDF file?"

**Strong:** "Write me a Python script, using Jupyter and xarray, to find
the maximum temperature of a netCDF file. The file is very large and may
need extra memory management to load. The script should have multiple
cells: one for loading libraries and setting file paths, one for defining
functions, and one where the functions are called. Number the cells for
easy future reference."

Same underlying question — the second version is checkable, reproducible,
and tells the model about a real constraint (file size) it would otherwise
never guess.

### Use commands, not questions
This one runs against social instinct, so it's worth explaining why.
- In **parenting**, a question offers options: "We're going to get in the
  car now, okay?"
- In **teaching**, you sometimes ask a question you don't want answered:
  "Why did you do that?" is really a prompt for reflection.
- With an **AI**, a question *necessitates an answer*, not a considered
  response — you spend the model's "effort" on answering the literal
  question asked, rather than on doing the underlying task well.

Prefer explicit commands: not "Could you maybe look at whether this
function is slow?" but "Analyze this function and explain why it may be
slow."

**Reminder:** the response you get back is a series of probability
distributions over next tokens — it isn't "talking to you" in the sense of
weighing your intent. Be as explicit as the task requires; don't make the
model guess what you meant.

### Applying this to common research tasks
- **Paper summaries:** extract the research question, methods, and data;
  synthesize findings and limitations; ask for page/section locations for
  deeper reading; then check numbers, units, citations, and figure
  references yourself, and confirm which claims are paraphrased vs.
  verbatim from the source.
- **Paper review:** ask the model to read like a journal editor — flow and
  clarity are things an LLM is genuinely good at judging, since it's
  effectively checking "does this read like something coherent." It's also
  good at telling you whether a secondary point you're considering has
  already been addressed elsewhere in the draft.
- **Grant alignment:** map your aims and sections against the actual
  solicitation text (paste it in — don't rely on the model's memory of
  funder rules), and ask it to flag missing requirements or narrative
  gaps. For long or complex solicitations, run this as several smaller,
  focused queries rather than one giant one — it's more reliable and
  easier to check each piece. The investigator, not the model, remains
  responsible for accuracy, originality, and compliance.

## Hands-on exercise
**Task — repair the prompt:** You've been given a netCDF file of ocean
data from 1959. Start from this weak prompt: *"Explain this model
output."* Rewrite it by adding:
- the variable(s), units, and scientific context
- the intended audience, the decision it's meant to inform, and the
  relevant plot or dataset
- what the model must verify, calculate, or flag

**Success looks like:** a rewritten prompt specific enough that two
different people using it would get comparably useful, checkable results.
For comparison, here's one strong version: *"I have a netCDF file of ocean
data for the year 1959. The file has physical tracer values as well as
zooplankton values. Write a Jupyter notebook to analyze the values and
give the yearly trend for each of the tracers. The results need to be
plotted suitable for professional publication, including PDF and PNG
formats. One plot should compare the trends of the three different
zooplankton types."* Compare what context your version added versus this
one — there's more than one good answer here.

## Key takeaways
- Specify goal, context, output, and boundaries — vague prompts get
  generic output
- Write full, grammatical sentences and explicit commands — LLMs aren't
  search engines and questions waste effort on answering rather than doing
- Use the five-part checklist (goal, evidence, output, check, confirm) for
  anything beyond a one-off question
- "Best" isn't just more detail — it's the version that's easiest to
  verify afterward
- For grant/policy work, paste in the actual source document rather than
  relying on the model's memory of it, and break complex checks into
  smaller queries
