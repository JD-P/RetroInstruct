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
