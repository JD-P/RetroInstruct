import json
from tqdm import tqdm
from weave import generate_outputs_vllm

with open("jdp_lorem_ipsum.txt") as infile:
    prompt = infile.read()

texts = []
pbar = tqdm(total=10000)
while len(texts) < 10000:
    texts += generate_outputs_vllm("mistralai/Mixtral-8x22B-v0.1", prompt, 2048, n=128, port=5001)
    pbar.update(128)
    if len(texts) % 2048 == 0:
        with open("lorem_ipsum.json", "w") as outfile:
            json.dump(texts, outfile)
    
with open("lorem_ipsum.json", "w") as outfile:
    json.dump(texts, outfile)
