import json
import random

with open("train.json") as infile:
    diffs = json.load(infile)

diff_formats = ["gnudiff", "git", "dmp"]

for diff in diffs:
    diff_format = random.choice(diff_formats)
    if diff_format == "gnudiff":
        instruction = diff["gnudiff_instruction"]
        diff_text = diff["gnudiff"]
    elif diff_format == "git":
        instruction = diff["gitdiff_instruction"]
        diff_text = diff["gitdiff"]
    elif diff_format == "dmp":
        instruction = diff["dmpdiff_instruction"]
        diff_text = diff["dmpdiff"]
    print(instruction, end="\n\n")
    print("<passage>")
    print(diff["text_corrupted"])
    print("</passage>", end="<|end|>")
    print("<diagnosis>")
    print(diff["operations"], end="")
    print("</diagnosis>")
    print("<diff>")
    print(diff_text, end="\n")
    print("</diff>")
    print("<repaired>")
    print(diff["text_clean"])
    print("</repaired>")
