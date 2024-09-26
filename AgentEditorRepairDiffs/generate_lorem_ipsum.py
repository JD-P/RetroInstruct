import asyncio
import random
import json
from tqdm import tqdm
from weave import generate_outputs_vllm

with open("jdp_lorem_ipsum.txt") as infile:
    ipsum = infile.read()
    posts = [i.strip().replace("\n</post>", "") for i in ipsum.strip().split("<post>\n") if i]

async def do_generate_outputs_vllm():
    random.shuffle(posts)
    prompt = '\n</post>\n\n'.join(["<post>\n" + post for post in posts]) + "\n</post>\n\n<post>"
    return generate_outputs_vllm("mistralai/Mixtral-8x22B-v0.1",
                                 prompt,
                                 2048,
                                 n=512,
                                 stop=["</post>"],
                                 port=5001)

async def main():
    num_connections = 0
    batches = []
    texts = []
    pbar = tqdm(total=120000)
    while len(texts) < 120000:
        while num_connections < 512:
            batches.append(do_generate_outputs_vllm())
            num_connections += 512
        texts += [text.strip() for text in await batches.pop()]
        num_connections -= 512
        pbar.update(512)
        if len(texts) % 2048 == 0:
            with open("lorem_ipsum.json", "w") as outfile:
                json.dump(texts, outfile)

    with open("lorem_ipsum.json", "w") as outfile:
        json.dump(texts, outfile)

asyncio.run(main())
