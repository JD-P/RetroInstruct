import random
import json

with open("expansions.json") as infile:
    expansions = json.load(infile)

with open("corrections.json") as infile:
    corrections = json.load(infile)

entries = []
for expansion in expansions:
    entry = {}
    entry["analogical-translation"] = expansion["expansion"]["analogical-translation"]
    entry["critique"] = expansion["expansion"]["flaws"]
    entry["subject"] = expansion["trace"]["subject"]
    entry["position"] = expansion["trace"]["position"]
    entry["salient-features"] = expansion["trace"]["salient-features"]
    try:
        entry["reference-class"] = expansion["trace"]["reference-class"]
    except:
        entry["reference-class"] = expansion["trace"]["reference\nclass"]
    entry["prior-arguments"] = expansion["trace"]["prior-arguments"]
    entry["chosen-argument"] = expansion["trace"]["chosen-argument"]
    entry["differences"] = expansion["trace"]["differences"]
    entry["label"] = "No"
    entries.append(entry)

for correction in corrections:
    entry = {}
    entry["analogical-translation"] = correction["analogical-translation"]
    entry["critique"] = correction["corrections"]
    entry["subject"] = correction["subject"]
    entry["position"] = correction["position"]
    entry["salient-features"] = correction["salient-features"]
    entry["reference-class"] = correction["reference-class"]
    entry["prior-arguments"] = correction["prior-arguments"]
    entry["chosen-argument"] = correction["chosen-argument"]
    entry["differences"] = correction["differences"]
    entry["label"] = "Yes"
    entries.append(entry)    

random.shuffle(entries)

train = entries[:-58]
val = entries[-58:]

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)
