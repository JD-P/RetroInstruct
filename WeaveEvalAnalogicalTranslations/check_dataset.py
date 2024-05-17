import json

with open("train.json") as infile:
    train = json.load(infile)

for entry in train:
    assert type(entry["analogical-translation"]) == str
    assert type(entry["critique"]) == list
    if False in [type(item) == str for item in entry["critique"]]:
        print(entry)
        raise Exception("Found it!")
    assert type(entry["subject"]) == str
    assert type(entry["position"]) == str
    assert type(entry["salient-features"]) == list
    if False in [type(item) == str for item in entry["salient-features"]]:
        print(entry)
        raise Exception("Found it!")
    assert type(entry["reference-class"]) == str
    assert type(entry["prior-arguments"]) == list
    if False in [type(item) == str for item in entry["prior-arguments"]]:
        print(entry)
        raise Exception("Found it!")
    assert type(entry["chosen-argument"]) == str
    assert type(entry["differences"]) == list
    if False in [type(item) == str for item in entry["differences"]]:
        print(entry)
        raise Exception("Found it!")
