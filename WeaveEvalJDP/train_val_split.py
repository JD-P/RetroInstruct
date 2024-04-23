import random
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("questions")
args = parser.parse_args()

with open(args.questions) as infile:
    questions = json.load(infile)

passages = list(questions.keys())
random.shuffle(passages)

train_passages = passages[:-40]
val_passages = passages[-40:]

train = {}
for passage in train_passages:
    train[passage] = questions[passage]

val = {}
for passage in val_passages:
    val[passage] = questions[passage]

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)
