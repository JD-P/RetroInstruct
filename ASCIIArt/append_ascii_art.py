import json

with open("train.json") as infile:
    train = json.load(infile)

with open("val.json") as infile:
    val = json.load(infile)

with open("train_i2a_conversions.json") as infile:
    train_i2a = json.load(infile)

with open("train_aic_conversions.json") as infile:
    train_aic = json.load(infile)

ascii_train = []
for prompt in train:
    i2a_art = train_i2a[prompt["filename"]]
    aic_art = train_aic[prompt["filename"]]
    prompt["art_i2a"] = i2a_art
    prompt["art_aic"] = aic_art
    ascii_train.append(prompt)

with open("val_i2a_conversions.json") as infile:
    val_i2a = json.load(infile)

with open("val_aic_conversions.json") as infile:
    val_aic = json.load(infile)

ascii_val = []
for prompt in val:
    i2a_art = val_i2a[prompt["filename"]]
    aic_art = val_aic[prompt["filename"]]
    prompt["art_i2a"] = i2a_art
    prompt["art_aic"] = aic_art
    ascii_val.append(prompt)

with open("ascii_train.json", "w") as outfile:
    json.dump(ascii_train, outfile)

with open("ascii_val.json", "w") as outfile:
    json.dump(ascii_val, outfile)
