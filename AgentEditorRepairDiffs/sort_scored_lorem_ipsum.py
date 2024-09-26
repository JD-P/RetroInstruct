import json
import time
import random

with open("lorem_ipsum_scored.json") as infile:
    ipsum = json.load(infile)

ipsum_l = [(i, ipsum[i]["score"]) for i in ipsum]
ipsum_l.sort(key=lambda x: x[1])

out = []
for index, i in enumerate(reversed(ipsum_l)):
    if index > 2700:
        continue
    out.append({"text": i[0], "score": i[1]})
random.shuffle(out)

with open("lorem_ipsum_final.json", "w") as outfile:
    json.dump(out, outfile)
