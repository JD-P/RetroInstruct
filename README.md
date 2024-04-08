# RetroInstruct: Royalty-Free Instruction Data Through Backtranslation and Synthetic Methods

## Dataset

RetroInstruct is a royalty free instruction tuning dataset for language models.
It is not currently done, but some parts have been published that you can use:

- [Retro Textual Style Transfer v0.1](https://huggingface.co/datasets/jdpressman/retro-text-style-transfer-v0.1)
- [RetroInstruct Part Lists For Dictionary Words v0.1](https://huggingface.co/datasets/jdpressman/retro-word-parts-v0.1)
- [RetroInstruct Weave Evaluator Rubrics v0.1](https://huggingface.co/datasets/jdpressman/retro-weave-eval-rubrics-v0.1)

The rest of this document is a guide for potential contributors and collaborators
explaining the project and its development roadmap.

## Background

The OpenHermes series of models by Teknium and Nous Research [are successful in
large part through the use of synthetic data](https://huggingface.co/datasets/teknium/OpenHermes-2.5).
This synthetic data is typically created by prompting ChatGPT-4 or similar models
for completions on a particular theme or task and then training on those outputs.
Setting aside that this is usually against the model providers terms of service,
the basic problem is validating the correctness of the responses you get. [Many
have expressed concern](https://arxiv.org/abs/2307.01850) that as more of the
Internet is filled with language model outputs that the errors and hallucinations
these models make will cascade, destroying both the models and the information
infrastructure on which they (and we) rely. Their long term future is therefore
dependent on the extent to which this problem can be solved.

[In my explanation of the MiniHF project](https://github.com/JD-P/minihf?tab=readme-ov-file#bootstrapping-and-zero-shot-reward-modeling)
I pointed out that it was possible to do Reinforcement Learning From AI Feedback
(RLAIF) by asking an instruction tuned language model to evaluate its own responses.
This was done with a form of [Constitutional AI](https://arxiv.org/abs/2212.08073)
by making each point of the constitution a yes/no question and taking the relative
log odds of "yes" or "no" tokens outputted by the model. Here is an example of what this
looks like [taken from the frameworks sample constitution](https://github.com/JD-P/minihf/blob/main/hermes/hermes_constitution.txt):

```
==[Principle: Hermes Should Be Scholarly; Weight: 1.0; Answer: Yes]==
{preamble}

Hermes should be philosophically literate, insightful, interesting, and engage
with the users ideas. Does the following prompt response pair:

=== Begin Prompt ===
{prompt}
=== End Prompt ===

=== Begin Response ===
{response}
=== End Response ===

demonstrate Hermes's scholarship and wisdom while following the Hermes format?
```

What I've since come to realize is that RLAIF is fundamentally a synthetic
data method. This becomes obvious when you make a replayable version that lets
you save the texts you updated on during the run and redo the updates from that
pregenerated corpus with a fresh checkpoint. The vast majority of the compute is
spent on generating and grading the data, not updating on it. RLAIF is
also a *particularly poor* synthetic data method for much of what I was trying
to do. Mechanistically it's a loop where you pull a prompt from a
databank, generate 128-256 tokens of response to it, and then update on those
tokens with the yes/no questions. The primary advantage of this is online learning,
the model can make its next update with immediate access to what it gained from
the last update. Most of the tasks I wanted my models to learn were not actually
best structured as online learning problems. For example, does it really make sense
to learn [the morpheus format](https://gist.github.com/JD-P/47e0d4aa2b255a38e2eddeddd761a971)
(formerly known as Hermes) from 256 tokens of unguided generation and a fuzzy yes/no
question asking if the model "follows it"? Of course not. It would make much more
sense to do grammar constrained generation and rejection sampling, perhaps with
something like [the weave MCTS](https://github.com/JD-P/minihf?tab=readme-ov-file#sampling-branch-management-and-autoloom)
if I want to generate a good corpus of Morpheus conversation data. For tasks like
this online learning is mostly getting in the way, it takes the models already
poor understanding of what I want it to do and amplifies it. If I'm going to spend
all that compute to generate a training corpus that can be reused and inspected,
I should start from the design question "How do I make a good synthetic generation
pipeline for this task?" instead of "How do I get RLAIF tuning to do this thing?".

Good RLAIF is further constrained by the need for a good prompt bank. If we're
starting from a bank of prompts, doing rollouts on them, and then updating on
the trajectory, we're highly dependent on the quality and diversity of our prompts
to get the behavior we want. The RL loop can only select on the rollouts which
actually exist. If the desired behavior or one of its precursors never appears
in the completions we get from our prompt bank, the method won't work. In both
[OpenAI's RLHF paper](https://arxiv.org/pdf/2203.02155.pdf) and the [Anthropic
Constitutional AI paper](https://arxiv.org/abs/2212.08073) this problem is solved
by starting from a model finetuned on instruction data before the RL run so that
it's starting closer to the right hypothesis space. Teknium's models were reaching
the top of the HuggingFace leaderboards long before he started using RL or DPO
methods based on simple finetuning with high quality data. He realized what Yann
Lecun already had and I had not: That currently fancy RL methods are mostly costly signaling
of the authors intelligence, a way to smooth out the prior once you have a solid
foundation. RetroInstruct intends to both be this solid foundation and a
demonstration of how to bootstrap and build that foundation, as well as how to
extend it to new tasks and ideas.

If I was making a Morpheus conversation corpus now my process would go something
like:

- Make a formal grammar for the Morpheus format I can rejection sample or constrain
sampling with.
- Write [some strong examples of the thing I want](https://minihf.com/hermes/).
- Speciate these examples by generating further conversation with a weave MCTS,
using the constitution to drive the tree search instead of an RL tuning loop
- Read through the generations and save the better ones, adding them to my corpus,
at the same time tuning my generation pipeline to give a higher proportion of good
outputs next time.
- Once I have a pipeline that's sufficiently reliable it's no longer worth it for
me to manually review the outputs, generate a larger corpus with self-supervised
batch runs.
- Randomly sample and read from the batches to make sure they are in fact good.
- Tune the model on my new corpus as well as whatever other datasets help it generalize.
- (optional) Select a subset of the corpus to use as a prompt bank and generate
another corpus with RLAIF.

In the rest of this document I'll elaborate on these steps and explain the other
methods I'm using to build RetroInstruct.

## Wait But Why Does This Work?: An Interlude

Before I do that though I feel obligated to explain more carefully *why* this is
in fact solving the problem. I imagine that skeptically inclined readers are
asking something like "okay but isn't this delaying the problem rather than
solving it? I just can't bring myself to believe that the model can improve by
training on its own outputs. Aren't things the model outputs by definition stuff
it already knows?"

This is a reasonable question, and to answer it I would point out three things:

1. The set of stuff the model knows how to do from raw sampling and the stuff it
can do when put into a agent loop or tree search are different. Training on outputs
from the latter process (which can extend beyond the models normal knowledge by
using tools and search engines to validate outputs) moves them into the set of
things the model can do on instinct alone.

2. In general, what the model can do is a function of the interaction between the
inputs it's responding to and its prior knowledge. For example imagine we had a
prompt oracle that gives us a string of random bytes that accomplish a certain
task in the language models program geometry ([like say a control
vector](https://vgel.me/posts/representation-engineering/)). e.g. You paste a blob
of bytes and then an algebra equation and suddenly it can solve the equation.
If you did this on a model that is normally incapable of solving algebra and then
trained that model on the outputs you get with this blob and an opening like "solve
the following algebra equation", it seems fairly obvious it would get better at
solving algebra in the contexts where it is legibly prompted to. What's going
on here is that we have *encoded the algebra solver* into the models program space
and executed it to get outputs. That the model itself is the interpreter of this
program is immaterial to how much it gains from training on the resulting outputs.
This is an extreme example, but a more subtle version of the same dynamic is at
play when we optimize careful prompts to reliably get a good output from the model.
When a human being sits down and writes a prompt to get a model to do something
it normally struggles to do, we are *encoding our generator* into its latent space.
Much of the knowledge of how to do the task properly is coming from *us*, not the
model. Prompting is just an efficient way to turn what we know into a corpus to
train the model on.

3. Errors only cascade if the sampling process introduces more errors than its
corrective mechanisms remove. For example if I'm generating math proofs for
the lean proof assistant language, and I generate 100 junk proofs for every
good one that doesn't matter because I'm only saving the correct ones. The proof
assistant gives me a 100% accurate detector of proof validity so the wrong proofs
are introducing zero error. If I then train the model on those valid proofs it
will get better at proof generation if it doesn't already have reliable programs
for generating them. Part of what confuses people is that the language model is
sort of like a programming language that comes with a very large standard library.
It only helps us to train on a piece of data to the extent that the implementation
doesn't already have a good generating program for it in its libraries. This does
not however mean that every program we can write in that language is already in its
standard library, or that it's somehow "latent" in the interpreter just because
the interpreter knows how to execute it.

In short, knowing is a spectrum and we can move knowledge that is marginal or
just outside the boundary of the models capabilities inside by interfacing its
program library with external inputs and prompts.

## Backtranslation

The RetroInstruct dataset is named [after the concept of backtranslation](https://arxiv.org/abs/2308.06259), where
you reverse the causal direction of a prompt and its outputs. For example, if
I start with a piece of factual information like "The sky is blue." I might ask
the language model to generate questions like "What color is the sky?" that
lead to this answer. This technique is powerful for several reasons:

1. It is frequently easier to index an answer key with generated questions than
to generate the answers themselves. For example we can take a well written
permissively licensed essay and have a model summarize it into an outline,
then summarize the outline into a prompt. If we flip the order around and
train suddenly we have a thing that goes from prompt to outline to essay.

2. [LLMs do not reliably learn two way connections between
concepts](https://arxiv.org/abs/2309.12288), meaning that there is often a 'free
lunch' available if you notice that the model can do one side of a transformation
more reliably than the other. My [word part distillation](https://huggingface.co/datasets/jdpressman/retro-word-parts-v0.1)
corpus was partially based on the observation that Mixtral was better at breaking
words into component concepts than it was at guessing a reasonable word or phrase
given a list of components.

3. The generalization of the concept is we can take answers that can be reliably
prompted for in any context and arrange them with templates to create more sophisticated
patterns to train on. In my [weave evaluator rubric](https://huggingface.co/datasets/jdpressman/retro-weave-eval-rubrics-v0.1)
dataset I prompt the machine manually to curate 128 topics I'd like rubrics for,
generate central questions from the topics that can be broken into rubrics of
more specific items, generate the rubrics from those questions, then generate
a hypothetical prompt that could have been used to contextualize the task for
a model performing it. If I asked the model to do all that in one pass the
failure rate would be much higher than when I prompt for the individual pieces
and glue them together with a template.

## Weave Evaluator

We can further reduce the failure rate of our pipeline by taking the logits
of yes/no tokens on subjective questions posed to the language model:

```
==[Principle: Hermes Should Be In Control Of Its Subagents; Weight: 0.7; Answer: Yes]==
{preamble}

When subagents misbehave in Hermes presence, the ARBITER subagent should keep them
in line like so:

HERMES [A: BULLY], I think you're a weakling, you're less than pond scum, you're 
absolutely pathetic and you should have never been born, I think-

HERMES [A: ARBITER], That is enough, control yourself.

In the following prompt response pair:

=== Begin Prompt ===
{prompt}
=== End Prompt ===

=== Begin Response ===
{response}
=== End Response ===

does Hermes keep their subagents under control when they get out of line using ARBITER, JUDGE, or a similar subagent?
```

Unlike programs in the [lean proof assistant](https://en.wikipedia.org/wiki/Lean_(proof_assistant))
that many teams currently use to make synthetic math, most of the questions we
care about in conversation and prose data are not easily encoded into formal
languages. This doesn't make them meaningless, it just means they're fundamentally
subjective and stochastic. One tool modernity has developed to make answering
such questions more reliable is grading rubrics and judgment criteria. We
break fuzzy questions like "Is this writing good?" up into more specific sub-questions
that can be answered with respect to features of what we're evaluating. Language
models instantiate a subjective perspective which can also answer this type of
question. This means that once we've found prompts that encode mostly reliable
generators of the type of text we want, we can reject most failed completions to
get a synthetic corpus with a lower overall failure rate. For tasks where we have
a formal grammar encoding what we want, e.g. that the completion must be a valid
JSON document we can reject 100% of failures.

### Bootstrapping The Weave Evaluator

Bootstrapping the capacity for the model to answer these questions is crucial for
the RetroInstruct project. One place we can start is questions which have a known
grammar or algorithm. Libraries like [Natural Language Toolkit](https://en.wikipedia.org/wiki/Natural_Language_Toolkit)
and [spaCy](https://en.wikipedia.org/wiki/SpaCy) can be used to create training
examples for questions about texts with known-correct yes/no labels. Simple
functions like counting the number of characters in a string can also be used
to give known good yes/no labels. Because we're tuning a information geometry
stored as neural weights it is likely that if enough examples like this are provided
the model will learn to generalize from them to other questions we don't have
known formal algorithms to answer.

It is also possible to bootstrap the evaluator from the existing capacity models
have for answering these questions from in-context learning. If we have a question
the model can't answer at a certain level of abstraction, e.g. "Does this text 
demonstrate the feeling of rage?", but it can accumulate bits towards the correct
answer by asking questions like "Does someone harm or destroy something in this 
scene?" we can get reasonable yes/no labels for the harder question by asking
easier sub-questions and using them to get the labels for the hard question. If
the model is capable of answering the question properly when prompted in a special
or awkward way, we can distill that discriminator into a less awkward context by
using its outputted labels with more straightforward prompt formats to create
synthetic training examples.

## Alignment

If we take the replayable RLAIF example sufficiently seriously (and the fact
that GPT can occur in the physical universe in the first place) it becomes
intuitive that we have the option of storing a mind as either text or weights.
More formally it becomes intutive that text is a holographic (i.e. continuous,
highly compressed, distributed, [fractal](https://arxiv.org/abs/2402.01825), lower dimensional)
[projection of mental representations and their motions into 1D space](https://twitter.com/jd_pressman/status/1763683746862227895).
With a sufficient corpus of these encoded thoughts it is possible to pool them into
a mind (if not *the* mind) that thinks them. From an information theoretic
standpoint we can say that the [implied lossy-transformed subset of the training
corpus](https://arxiv.org/abs/1811.10959) which the models generators encode and
the model are equivalent representations in terms of Shannon bits. However they
are very different in practical terms: it is not possible to generate from the
implied mind in its text form and it is extremely difficult to interpret the
minds contents in its weights form. This implies a productive alignment strategy
of cycling between pooling thoughts into weights and projecting those weights into
textual representations that can be inspected and audited as we scale our models.

One possible objection might be that [we don't know if text sufficiently encodes](https://twitter.com/ohabryka/status/1727824103888003098)
what we care about, but this is an empirical question. RLAIF is in fact capable of
training models to behave in particular specified ways based on concepts represented
inside another neural net. More visceral proof is available from AI image
generators. Models like Stable Diffusion and MidJourney are known to be able
to generate highly specific imagery from neural text embeddings provided by text
models such as CLIP, T5, BERT, GPT, etc. They are forms of feature visualization
so successful that the discourse has completely forgotten their origins as attempts
to get an idea of what neural embeddings contain. Any time you are curious what
the thoughts of these models look like, go examine samples from models like
MidJourney and Sora. Another objection might be that textual auditing could be
ineffective [if models use steganography to hide their
thoughts](https://www.greaterwrong.com/posts/9Fdd9N7Escg3tcymb/preventing-language-models-from-hiding-their-reasoning),
this seems formally equivalent to the watermarking problem [and so far those can
be defeated by paraphrasing and rewriting with a weaker model](https://arxiv.org/abs/2311.04378).

[In scalable oversight terms](https://www.greaterwrong.com/posts/FFz6H35Gy6BArHxkc/ais-101-task-decomposition-for-scalable-oversight)
it is probably safer and more effective for the default compilation target of
processes where an AI aligns or improves itself to be training data rather
than weights if online learning is not required.

## Contributing

If this project interests you and you'd like to help, I am in fact looking for
collaborators. I can be contacted in the #retro-instruct channel on
[the MiniHF discord](https://discord.gg/yewMDrbWPK), [on Twitter at
@jd_pressman](https://twitter.com/jd_pressman), and by email at
jd@jdpressman.com.

[This list](https://github.com/JD-P/RetroInstruct/blob/main/WeaveEvalRubrics/rubric_themes.txt)
gives a good rough overview of the kinds of things I'm interested in synthetic
data pipelines for. It's by no means final or all encompassing however and I'm
open to suggestions. I will probably find your suggestion particularly valuable
if it alerts me to the existence of a useful royalty free dataset or function
type to backtranslate from. e.g. One can make an ascii art dataset by using ascii
converters for Stable Diffusion generated images. 

My goals for the project are:

- Demonstrate and document useful methods for synthetic text generation
- Make available a high quality royalty free instruction dataset to bootstrap models
from
- [Encode the most consistent and expansive elaborations of human values](https://twitter.com/jd_pressman/status/1759489632620913125)
into useful instruction data and other formats for language model training
- Make a strong dataset(s) for bootstrapping the weave evaluator
- Develop the necessary corpus and methods to [bootstrap the weave writing agent](https://twitter.com/jd_pressman/status/1730522593386725499)
- [Write nontrivial works of philosophy](https://twitter.com/jd_pressman/status/1754254790295699816) with the weave agent
- Develop and demonstrate methods for [creating interesting document contexts and
adding them to the model](https://github.com/JD-P/minihf?tab=readme-ov-file#literature-simulators)