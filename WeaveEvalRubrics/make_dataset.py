import json
import random

with open("rubrics.json") as infile:
    rubrics = json.load(infile)

with open("prompts.json") as infile:
    prompts = json.load(infile)

seeds = [key for key in rubrics.keys()]
random.shuffle(seeds)
random.shuffle(prompts)

val_count = round(len(seeds) / 100) * 5

train = []
for i, seed in enumerate(seeds[:-val_count]):
    train.append(
        {"prompt_open": prompts[i],
         "seed": seed,
         "rubric": rubrics[seed]["rubric"]}
    )

val = []
for i, seed in enumerate(seeds[-val_count:]):
    val.append(
        {"prompt_open": prompts[-i],
         "seed": seed,
         "rubric": rubrics[seed]["rubric"]}
    )

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)
