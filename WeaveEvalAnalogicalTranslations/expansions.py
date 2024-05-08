import json
import random
from tqdm import tqdm
from argparse import ArgumentParser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

parser = ArgumentParser()
parser.add_argument("api_key")
parser.add_argument("prompt")
parser.add_argument("--temp", default=1, type=float)
parser.add_argument("-o", "--output", default="expansions.json")
parser.add_argument("-r", "--resume")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

with open("analogies.json") as infile:
    analogies = json.load(infile)
    
with open(args.prompt) as infile:
    prompt = infile.read()

if args.resume:
    with open(args.resume) as infile:
        expansions = json.load(infile)
else:
    expansions = {}

for subject in tqdm(analogies.keys()):
    if subject in expansions:
        continue
    entries = []
    for entry in analogies[subject]:
        messages = [
            # ChatMessage(role="system", content=system_prompt),

            ChatMessage(role="user", content=prompt.replace("REPLACE_WITH_ENTRY", json.dumps(entry)))
        ]

        chat_response = client.chat(
            messages,
            model,
            temperature=args.temp,
            response_format={"type": "json_object"},
        )

        try:
            # Make sure we got output in expected format
            response = chat_response.choices[0].message.content
            response = response.replace("\\n", "\n")
            response = response.replace("\n", "\\n")
            expansion = json.loads(response)
            assert type(expansion) == dict
            len(expansion) == 2
            assert "analogical-translation" in expansion
            assert type(expansion["analogical-translation"]) == str
            assert "flaws" in expansion
            assert type(expansion["flaws"]) == list
        except:
            # Give it another try
            print(expansion)
            print(f'"{subject}" failed. Retrying..."')
            chat_response = client.chat(
                messages,
                model,
                temperature=args.temp,
                response_format={"type": "json_object"},
            )
            response = chat_response.choices[0].message.content
            response = response.replace("\\n", "\n")
            response = response.replace("\n", "\\n")
            expansion = json.loads(response)
        entries.append({"trace":entry, "expansion":expansion})
    # Store it this way so it's easy to skip if we have to resume
    expansions[subject] = entries
    if len(expansions) % 5 == 0:
        with open(args.output + ".wip", "w") as outfile:
            json.dump(expansions, outfile)
    
with open(args.output, "w") as outfile:
    final = []
    for subject in expansions:
        for entry in expansions[subject]:
            final.append(entry)
    json.dump(final, outfile)
