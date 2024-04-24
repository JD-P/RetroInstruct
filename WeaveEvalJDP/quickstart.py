import json
import random

# Load yes or no prefix strings
with open("yes_or_no.txt", "r") as file:
    yes_no_prefixes = [line.strip() for line in file.readlines()]

# Load the data
with open("train.json") as infile:
    data = json.load(infile)

mistral_instruction = "<s> [INST]{}[/INST]{}"

for row in data:
    passage = row["passage"]
    questions = row["questions"]
    random.shuffle(questions)
    
    # Choose a random yes/no prefix
    yes_no_prefix = random.choice(yes_no_prefixes)

    instruction = f"{yes_no_prefix} In the following passage:\n\n<passage>\n{passage}\n</passage>\n\n"
    output = ""

    for i, question in enumerate(questions):
        question_text = question["question"]
        label = question["label"]
        explanation = question["explanation"]

        if i < len(questions) - 1:
            instruction += f"{question_text} {label}. {explanation}\n\n"
        else:
            instruction += f"{question_text}"
            output += f"{label}. {explanation}"

    print(mistral_instruction.format(instruction, output))


