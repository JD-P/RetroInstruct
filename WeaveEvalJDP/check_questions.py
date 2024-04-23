import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("questions")
args = parser.parse_args()

with open(args.questions) as infile:
    questions = json.load(infile)

for passage in questions:
    # assert len(questions[passage]) == 5
    for question in questions[passage]:
        try:
            assert "type" in question
            assert "question" in question
            assert "label" in question
            assert "explanation" in question
            assert question["label"].lower() in {"yes", "no"}
        except AssertionError:
            print(question)
