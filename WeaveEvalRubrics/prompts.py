import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

parser = ArgumentParser()
parser.add_argument("api_key")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

with open("rubrics.json") as infile:
    rubrics = json.load(infile)

with open("prompts_prompt.txt") as infile:
    prompts_prompt = infile.read()


prompts = []
seed = 2558
progress = tqdm(total=2560)

for i in range(250):
    if len(prompts) >= 2560:
        break
    messages = [
        ChatMessage(role="user", content=prompts_prompt)
        ]

    chat_response = client.chat(
        messages,
        model,
        temperature=1,
        random_seed=seed,
        response_format={"type": "json_object"},
    )

    
    try:
        openings = json.loads(chat_response.choices[0].message.content)
        # Make sure we got output in expected format
        assert type(openings) == dict
        len(openings) == 1
        for opening in openings["openings"]:
            assert type(opening) == str
            assert "{seed}" in opening
            assert not opening.startswith(".")
            assert not opening.startswith(">")
            assert not opening.startswith(";")
            assert not opening.startswith(":")
    except:
        print("Failed. Iterating seed...")
        seed += 1
        continue
    # Store it this way so it's easy to skip if we have to resume
    prompts += openings["openings"] 
    if len(prompts) % 100 == 0:
        with open("prompts.json", "w") as outfile:
            json.dump(prompts, outfile)
    seed += 1
    progress.update(len(openings["openings"]))
    
with open("prompts.json", "w") as outfile:
    json.dump(prompts, outfile)
