import os
import json
import random

with open("images.json") as infile:
    images = json.load(infile)

pool = []
    
for prompt_triplet in images:
    subject, style, prompt = prompt_triplet.split("|")
    filename = images[prompt_triplet] + ".png"
    if os.path.exists(filename):
        pool.append({"subject":subject,
                     "style":style,
                     "prompt":prompt,
                     "filename":filename})
    else:
        print(f"'{filename}' for prompt {prompt} does not exist")

random.shuffle(pool)

train = pool[:-300]
val = pool[-300:]

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)
