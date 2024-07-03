import json
import time

with open("lorem_ipsum_scored.json") as infile:
    ipsum = json.load(infile)

ipsum_l = [(i, ipsum[i]["score"]) for i in ipsum]
ipsum_l.sort(key=lambda x: x[1])

for index, i in enumerate(reversed(ipsum_l)):
    if i[1] > 2.3:
        continue
    print(index, i[1], i[0], "[END]", end="\n\n")
