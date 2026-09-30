# Module 01: How AI Works

## Objectives
By the end of this module you will be able to:
- Correctly place "AI," "machine learning," and "deep learning" relative to
  each other, and explain why AI is a statistical tool, not a mind
- Describe, at a mechanical level, how an LLM turns your prompt into a
  response (tokens → embeddings → attention → next-token prediction)
- Identify whether a research task calls for deductive, inductive, or
  abductive reasoning — and why that matters for what to trust from an LLM
- Correctly identify what kinds of tasks LLMs are strong vs. unreliable at
- Recognize hallucination and know when to independently verify output

## Why this matters in a research context
Treating an LLM like a search engine or a calculator leads to two failure
modes: over-trusting fabricated facts, or under-using a tool that's
genuinely good at drafting, summarizing, and restructuring. Knowing the
mechanism — and knowing which reasoning mode you're actually asking it to
do — fixes both.

## Core concept

### AI, ML, and DL aren't three things — they're nested
- **Artificial Intelligence** — the broad idea of a computer behaving like
  a human in some respect.
- **Machine Learning** — a *subset* of AI: using math (mostly linear
  algebra) to find connections between data points.
- **Deep Learning** — a *subset* of ML: using machine learning itself to
  set the parameters of the math that finds those connections.

Two things follow directly from this that are worth internalizing before
anything else:

- **AI is statistical.** It finds patterns in data and returns the most
  likely continuation — it is not consulting a fact database.
- **AI is a tool, not a mind.** It has no understanding, no goals, and no
  judgment of its own. Many research misconceptions about AI come from
  anthropomorphizing it — treating "it sounds confident" as "it is
  right."

### Predictive vs. generative
- **Predictive models** identify statistical patterns between inputs and
  known outcomes — e.g. training a model to estimate the evapotranspiration
  parameters of large trees from environmental data (PyTorch, Keras).
- **Generative models** (what you're using when you chat with an LLM)
  produce new content — text, code, images — by sampling from a learned
  distribution.

**Temperature** controls the randomness of that sampling: low temperature
(0–0.3) gives deterministic, factual-leaning output; medium (0.5–0.7)
balances creativity and coherence; high (0.8–1.0) is more creative and
diverse but less reliable. Use low temperature for math/facts, higher for
brainstorming or creative writing. (Not every interface exposes this
setting, but it's worth knowing it exists.)

### How a prompt actually becomes a response
1. **Tokenization** — your text is chopped into small pieces called
   tokens, and each token gets a unique number. `"What color is the sky?"`
   becomes roughly `["What", " color", " is", " the", " sky", "?"]`, each
   mapped to an integer id.
2. **Embeddings** — each token id is swapped for an embedding vector (a
   list of hundreds to a couple thousand numbers) that encodes how close
   that token is, in meaning, to every other token the model knows.
3. **Attention** — for every token, the model asks *which other tokens
   are relevant to understanding this one* and learns to weight them
   accordingly. This is what lets the model track "it" back to the right
   noun, or a variable back to its definition three paragraphs earlier.
4. **Prediction** — at inference time, the model is **autoregressive**: it
   predicts one token at a time, and each predicted token gets fed back in
   as context for predicting the next one. `"The"` → predicts `"sky"` →
   (now with "The sky" as context) predicts `"is"` → predicts `"blue"` →
   predicts `"."`.

The practical upshot: the model is not retrieving an answer, it's
generating one token at a time based on what's statistically likely to
follow — for every single word, including the ones inside a fact or
citation.

### Three ways of reasoning — and which one your task needs
Understanding which mode of reasoning a task requires tells you how much
to trust the output.

| Type | Direction | Pattern | Example | Certainty |
|---|---|---|---|---|
| **Deductive** | General → specific | Start from a rule known to be true, apply it to a case, reach a guaranteed conclusion | "All mammals breathe air. Dolphins are mammals. ∴ Dolphins breathe air." | Certain, *if* premises are true |
| **Inductive** | Specific → general | Observe many examples, generalize a likely rule | "Every swan I've seen is white. ∴ Probably all swans are white." | Likely, not guaranteed |
| **Abductive** | Observation → best explanation | Observe something incomplete, generate candidate explanations, pick the simplest/most plausible | "The grass is wet. Rain is the simplest explanation. ∴ It probably rained." | A best guess, not a proof |

Why this matters for LLMs specifically: **your model was *trained*
inductively** (generalizing patterns from massive text data), **it
*executes* known rules deductively** when running code or a formal proof,
and **it reasons *abductively*** whenever you hand it a novel problem and
ask it to explain or diagnose something. Modern LLMs blend all three
depending on the task — but none of the three is "look up the true
answer." Code execution and formal logic are the closest thing to real
certainty you'll get from these tools.

### What AI can and cannot do

| Can't do | Why |
|---|---|
| Verify its own outputs | It generates plausible text, not verified facts — it has no internal check for "is this true" |
| Do real science | It cannot run experiments, collect data, or causally reason beyond patterns in its training |
| Cite reliably | Fabricated citations are extremely common — always verify every reference independently |
| Maintain true confidentiality | Inputs to cloud LLMs may be used for training — never upload sensitive or restricted data |
| Think or understand | It manipulates tokens statistically; there is no comprehension, intent, or consciousness |
| Keep up with new research | Models have training cutoffs — anything published after that is unknown to the model unless it searches |

What it *is* good at, correspondingly: transforming and drafting from
material you supply (summarizing, reformatting, restructuring), and acting
as a first-pass reviewer or editor on something you can independently
check.

### Types of LLMs
LLMs differ along several axes: training data, training language,
openness, size, and intended use case (a model meant to run on a phone vs.
one meant to run on a server are built very differently).

At a high level:
- **Closed / frontier models** (e.g. GPT, Claude, Gemini) — strong general
  reasoning, usually accessed via a web app or API, not self-hostable.
- **Open-weight models** (e.g. Llama, Mistral, Qwen, DeepSeek) — you can
  download and run these yourself; more control and privacy, but you need
  the hardware.
- **Domain-tuned models** (e.g. BioMedLM, Galactica) — narrower, tuned for
  a specific field.

You'll also see **MoE (Mixture of Experts)** models — instead of one huge
network, the model is built from many smaller "expert" sub-networks, only
some of which activate per query. This is why a model can list a huge
total parameter count but a much smaller "active" parameter count.

### A light word on responsible use
This module is about mechanics, not ethics, but three things are worth
flagging now (each of these could be its own deeper lesson later):
- **Hallucination** — covered above; treat every unverified factual claim,
  number, or citation as a draft to check, not a fact.
- **Bias** — training data overrepresents certain languages, cultures, and
  viewpoints; AI output can quietly encode those imbalances, which matters
  most in social-science, clinical, or policy-relevant work.
- **Disclosure and environmental cost** — many journals and funders now
  require disclosing AI use, and running these models has a real (if often
  overstated and highly variable) energy and water cost. Right-size the
  model to the task.

## Hands-on exercise
**Task:** Ask an AI assistant a factual question about a narrow, obscure
topic in your field (something you know well enough to fact-check)
without giving it source material. Then ask it to cite its sources. Then
identify: was the task you gave it deductive, inductive, or abductive in
nature?

Do this in [exercise.ipynb](exercise.ipynb).

**Success looks like:** You can point to at least one specific claim or
citation in the response that is either fabricated, subtly wrong, or that
the model can't actually substantiate — and you can explain *why* that
happened based on the mechanism above, and which reasoning mode the task
fell into.

## Try it: interactive pipeline demo

Want to see the mechanism above happen for real? There's a small in-browser
demo that runs an actual small language model (`distilgpt2`) on a sentence
you type, showing real tokenization and real next-token prediction (the
embeddings and attention panels are clearly-labeled illustrative
approximations — see the caption on each):

**[Open the interactive demo](https://jbrzensk.github.io/AI_for_SIO/modules/01-how-ai-works/demo/)**

First load takes a one-time ~100-300MB download (cached by your browser
afterward). No data leaves your browser — everything runs locally.

## Key takeaways
- AI ⊃ Machine Learning ⊃ Deep Learning — it's statistical, and it's a
  tool with no understanding, goals, or judgment of its own
- A prompt becomes a response through tokenization → embeddings →
  attention → autoregressive next-token prediction — there's no fact
  lookup anywhere in that pipeline
- Know which reasoning mode a task needs: deductive (rule-following, most
  trustworthy), inductive (pattern generalization, how the model was
  trained), or abductive (best-guess explanation, least certain)
- LLMs excel at transforming/drafting text you supply; they're unreliable
  for unverified facts, citations, or anything past their training cutoff
- Closed/frontier, open-weight, and domain-tuned models trade off
  capability, control, and hardware requirements differently
- Verify anything checkable that matters; don't outsource judgment on
  high-stakes claims
