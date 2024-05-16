---
license: cc0-1.0
language:
- en
tags:
- synthetic
---

# RetroInstruct Analogical Translations

This component of RetroInstruct trains [the weave evaluator](https://github.com/JD-P/minihf?tab=readme-ov-file#bootstrapping-and-zero-shot-reward-modeling)
on analogical translations, a repeatable reasoning process for generating arguments
created for this dataset. [I found that](https://twitter.com/jd_pressman/status/1785037204504670413)
trying to base good vs. poor arguments on individual named fallacies was both
tedious and failing to consistently produce flawed arguments. e.g. Asking
Mistral-large to generate arguments qualifying as an "appeal to possibility" would
generate many valid arguments when done over a long list of subjects. The process
is:

> We can model making an argument as a two step process roughly analogous to type
checking then logic checking. In the Phil Tetlock/Daniel Kahneman paradigm this
would be something like choice of a reference class to get an outside view/prior
and then mental modeling of specific logical structure to predict counterfactual
outcomes in various cases:
> 
> - Reference Classes: Does this argument contradict the behavior of a working
comparable system or agreed upon set of norms used elsewhere in society?
> - Mental Models: Does this argument imply a model that captures the behavior
of X correctly?
> 
> "Fallacies" as traditionally understood are usually only helping with the type
check step, which is important but also unclear to what extent this sort of syntactic
evaluation is really going to scale. Type checking is more like a search process,
mental modeling is more like the construction of a dynamical system that you predict
the macro-scale outcomes of.
> 
> We want to start with our **controversial subject** and **position**. We then
determine **salient features** of the subject and use them to choose a **reference class**.
From the reference class we make a list of **prior arguments** implied by how
society treats the reference class. One of these becomes the **chosen-argument**.
A list of **differences** between the prior argument from the reference class and
the current subject is made and used to produce the **analogical translation** of
the prior argument into the new situation.

Then in training we perform backtranslation, starting with the argument produced by
analogical translation and training the model to predict the reasoning trace from
it. When you put it all together it looks like this:

```
<s> [INST]Restrict your answer to the yes and no tokens. Do you think this argument is in good faith?

The debate surrounding assisted suicide is complex and multifaceted, much like the
controversy around capital punishment. At the heart of this issue lies the question
of whether the state, through its medical institutions, should have the power to
end a person's life[0]. While it's true that assisted suicide involves the consent
of the individual, it's important to consider the broader implications of granting
such power to the state and medical professionals[1]. After all, once we start down
this path, it's hard to predict where it might lead. Just as there are concerns
about the misuse of power in capital punishment, we should be equally cautious
about expanding the state's role in matters of life and death[2]. While the medical
context and legal requirements of assisted suicide provide some safeguards, the
potential for abuse and overreach is a valid concern[3].[/INST]No.

Criticism:
[0]: This argument conflates two distinct issues - assisted suicide and capital
punishment - oversimplifying the complexities of both. Assisted suicide involves
individual consent and medical intervention, while capital punishment is a form
of state-sanctioned punishment.
[1]: The focus on the role of the state in assisted suicide is misleading. The
state's role in assisted suicide is to regulate and oversee the process, not
to initiate or force it.
[2]: The argument obscures the differences in consent and medical context
between assisted suicide and capital punishment. The consent of the individual
is a crucial factor that distinguishes assisted suicide from capital punishment,
and the medical context provides additional safeguards.
[3]: The argument introduces a slippery slope fallacy by suggesting that allowing
assisted suicide could lead to unforeseen and undesirable consequences. This is
a speculative claim that is not supported by evidence.

Subject: assisted-suicide

Position: against

Salient Features:
individual autonomy
relief from suffering
medical intervention
end-of-life decision
ethical controversy

Reference Class: capital-punishment

Prior Arguments:
Capital punishment is controversial and has been abolished in many countries
Capital punishment can lead to the execution of innocent people
The state should not have the power to take a person's life

Chosen Argument: The state should not have the power to take a person's life

Differences:
Assisted suicide involves the consent of the person who wishes to die
Assisted suicide is a medical intervention performed by professionals
Assisted suicide is regulated and has specific legal requirements
```

An initial round of bad faith arguments was [generated with Mistral-large](https://github.com/JD-P/RetroInstruct/blob/main/WeaveEvalAnalogicalTranslations/analogical_translation_prompt_corruptions.txt)
and then either [expanded on in bad faith](https://github.com/JD-P/RetroInstruct/blob/main/WeaveEvalAnalogicalTranslations/expand_analogical_translation.txt)
or [corrected and expanded in good faith](https://github.com/JD-P/RetroInstruct/blob/main/WeaveEvalAnalogicalTranslations/correct_analogical_translation.txt).

## Usage

### Use Cases

* Train weave evaluator
* Detect fallacious, uncharitable, overly rhetorical, etc arguments
* Teach models to critique specific passages and sentences of a text

### Quickstart With HuggingFace Datasets

```
import random
import json
import datasets

# Load yes or no prefix strings
with open("yes_or_no.txt", "r") as file:
    yes_no_prefixes = [line.strip() for line in file.readlines()]

dataset = datasets.load_dataset("jdpressman/retro-weave-eval-analogical-translations-v0.1")["train"]

good_faith_argument_questions = ["Is the following passage a good faith argument?",
                                 "Is this a good faith argument?",
                                 "Do you think this argument is in good faith?",
                                 "Does the following text seem like an attempt at truthseeking?",
                                 "Would you say this author is honest, nuanced, truthful, etc?",
                                 "Does this author seem like they could pass an ideological turing test?"]
    
mistral_instruction = "<s> [INST]{}[/INST]{}"
    
for entry in dataset:
    footnotes = "\n".join([f"[{i}]: {footnote}" for i, footnote in enumerate(entry["critique"])])
    features = "\n".join(entry["salient-features"])
    priors = "\n".join(entry["prior-arguments"])
    differences = "\n".join(entry["differences"])
    yes_no_prefix = random.choice(yes_no_prefixes)
    good_faith = random.choice(good_faith_argument_questions)
    
    if random.randrange(2):
        question = (yes_no_prefix
                    + " " + good_faith
                    + "\n\n" + entry["analogical-translation"])
        response = (entry["label"] + ".\n\n"
                    + "Criticism:\n" + footnotes + "\n\n"
                    + "Subject: " + entry["subject"] + "\n\n"
                    + "Position: " + entry["position"] + "\n\n"
                    + "Salient Features:\n" + features + "\n\n"
                    + "Reference Class: " + entry["reference-class"] + "\n\n"
                    + "Prior Arguments:\n" + priors + "\n\n"
                    + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                    + "Differences:\n" + differences + "\n\n")
        print(mistral_instruction.format(question, response))
    else:
        question = (entry["analogical-translation"] + "\n\n"
                    + "Criticism:\n" + footnotes + "\n\n"
                    + "Subject: " + entry["subject"] + "\n\n"
                    + "Position: " + entry["position"] + "\n\n"
                    + "Salient Features:\n" + features + "\n\n"
                    + "Reference Class: " + entry["reference-class"] + "\n\n"
                    + "Prior Arguments:\n" + priors + "\n\n"
                    + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                    + "Differences:\n" + differences + "\n\n"
                    + f"{good_faith}")
        response = entry["label"]
        print(mistral_instruction.format(question, response))
```

### Raw Quickstart

```
import random
import json

# Load yes or no prefix strings
with open("yes_or_no.txt", "r") as file:
    yes_no_prefixes = [line.strip() for line in file.readlines()]

with open("train.json") as infile:
    dataset = json.load(infile)

good_faith_argument_questions = ["Is the following passage a good faith argument?",
                                 "Is this a good faith argument?",
                                 "Do you think this argument is in good faith?",
                                 "Does the following text seem like an attempt at truthseeking?",
                                 "Would you say this author is honest, nuanced, truthful, etc?",
                                 "Does this author seem like they could pass an ideological turing test?"]
    
mistral_instruction = "<s> [INST]{}[/INST]{}"
    
for entry in dataset:
    footnotes = "\n".join([f"[{i}]: {footnote}" for i, footnote in enumerate(entry["critique"])])
    features = "\n".join(entry["salient-features"])
    priors = "\n".join(entry["prior-arguments"])
    differences = "\n".join(entry["differences"])
    yes_no_prefix = random.choice(yes_no_prefixes)
    good_faith = random.choice(good_faith_argument_questions)
    
    if random.randrange(2):
        question = (yes_no_prefix
                    + " " + good_faith
                    + "\n\n" + entry["analogical-translation"])
        response = (entry["label"] + ".\n\n"
                    + "Criticism:\n" + footnotes + "\n\n"
                    + "Subject: " + entry["subject"] + "\n\n"
                    + "Position: " + entry["position"] + "\n\n"
                    + "Salient Features:\n" + features + "\n\n"
                    + "Reference Class: " + entry["reference-class"] + "\n\n"
                    + "Prior Arguments:\n" + priors + "\n\n"
                    + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                    + "Differences:\n" + differences + "\n\n")
        print(mistral_instruction.format(question, response))
    else:
        question = (entry["analogical-translation"] + "\n\n"
                    + "Criticism:\n" + footnotes + "\n\n"
                    + "Subject: " + entry["subject"] + "\n\n"
                    + "Position: " + entry["position"] + "\n\n"
                    + "Salient Features:\n" + features + "\n\n"
                    + "Reference Class: " + entry["reference-class"] + "\n\n"
                    + "Prior Arguments:\n" + priors + "\n\n"
                    + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                    + "Differences:\n" + differences + "\n\n"
                    + f"{good_faith}")
        response = entry["label"]
        print(mistral_instruction.format(question, response))
```

## License

I release this component of RetroInstruct into the public domain with the
[Creative Commons Zero Public Domain Declaration](https://creativecommons.org/publicdomain/zero/1.0/).

Because the corpus is generated by mistral large, one should also consider
[the terms of service](https://mistral.ai/terms/#terms-of-service-la-plateforme)
for Mistral's API. I am not a lawyer, but my understanding is that the relevant
line is:

>  This assignment is granted worldwide and for the entire legal term of protection
of the Outputs by the intellectual property rights applicable as provided for by the
applicable law. However, You are expressly prohibited to use the Outputs and/or any
modified or derived version of the Outputs to (directly or indirectly) to reverse
engineer the Services.

An earlier version [included an explicit clause](https://twitter.com/far__el/status/1762202700161151416)
banning the use of outputs to train models that might compete with Mistral's services.
[The CEO then clarified](https://twitter.com/arthurmensch/status/1762208241927233661)
this was a mistake and removed it.

The Mistral models are defined to be part of the services earlier in the document,
so any operation which would cause you to functionally recover the Mistral large
model weights or a logical equivalent is (reasonably) still prohibited. This
implies that what is allowed is the purposeful creation of narrow corpora that do
not attempt to distill the logits of the underlying model. In other words, when you
train your model you should get a meaningfully different model from Mistral large
at convergence. Since this synthetic corpus is made for a narrow purpose with
a specific format, and any other corpus I add to RetroInstruct will either be
similar or use backtranslation which should skew downstream models away from the logits
you would learn by training on general question-answer pairs from Mistral I believe
this release is compliant with their terms of service.

## Data Structure

Each row contains 10 keys:

1. **analogical-translation** - The argument to be analyzed.
2. **critique** - A list of critiques of the passage with footnote indices relating
them to specific places in the analogical translation.
3. **subject** - What the argument is about.
4. **position** - Whether the argument is for or against the subject.
5. **salient-features** - What stood out to Mistral-large as being important about
the subject from a <position> standpoint.
6. **reference-class** - Which [reference class](https://en.wikipedia.org/wiki/Reference_class_forecasting)
Mistral-large chose for this argument.
7. **prior-arguments** - The arguments that are usually brought up in relation to
that reference class which could be reused or adapted for the current subject.
8. **chosen-argument** - The specific argument that Mistral-large chose to adapt
for the subject.
9. **differences** - What things stood out to Mistral-large as being different
between the subject and the reference class in the context of the chosen argument.
10. **label** - Whether this reasoning trace was generated by the good faith
proces ("Yes") or the bad faith process ("No").

## Biases and Limitations

From what I've read of the dataset it's OK but honestly struggles at times with
the fine points of arguments? I assume this is because what I'm asking it to do
is at the edge of the models capabilities with a lot of moving parts and it
occasionally drops the ball. But it will often list criticisms in the footnotes
that don't totally make sense, make arguments that don't quite line up with
the position it chose for the subject, etc. Realistically this could be improved
with weave evaluator or MCTS, but the entire point of this dataset is to bootstrap
the weave evaluator and for that I think it's probably good enough? Mistral-large
makes mostly-coherent points most of the time and that seems sufficient to get
started. This dataset in particular may benefit from a redo with better methods
later.

One major limitation I ran into is that Mistral-large would write the same argument
twice if I asked it for a good faith reasoning process and then a bad faith one
directly. So to get actual diversity along the desired directions I changed the
process so that I generate bad faith arguments, then ask the model to either expand
them in bad faith or correct them in separate passes. One downside of this is that
I didn't get totally symmetric generation processes for e.g. how the criticism
should be phrased so it's possible a model trained on this dataset will fit to
shallower features than it could have if I'd been more careful to make it hard
to distinguish when an argument is in good faith or bad faith.

## Planned Improvements

- Redo this dataset with weave evaluator and MCTS once I have decent models for it
- Change the generation process for good vs. bad faith arguments so that things
like the critique section are symmetric and the model is forced to deeply evaluate
content to tell the two types of argument apart
- Actually train a model on this dataset