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
parser.add_argument("-o", "--output", default="corrections.json")
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
        corrections = json.load(infile)
else:
    corrections = {}

for subject in tqdm(analogies.keys()):
    if subject in corrections:
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
            correction = json.loads(response)
            assert type(correction) == dict
            # len(correction) == 2
            assert "subject" in correction
            assert type(correction["subject"]) == str
            assert "position" in correction
            assert correction["position"].lower() in ["for", "against"]
            assert "salient-features" in correction
            assert type(correction["salient-features"]) == list
            assert "reference-class" in correction
            assert type(correction["reference-class"]) == str
            assert "prior-arguments" in correction
            assert type(correction["prior-arguments"]) == list
            assert "chosen-argument" in correction
            assert type(correction["chosen-argument"]) == str
            assert "differences" in correction
            assert type(correction["differences"]) == list
            assert "analogical-translation" in correction
            assert type(correction["analogical-translation"]) == str
            assert "corrections" in correction
            assert type(correction["corrections"]) == list
        except:
            # Give it another try
            print(correction)
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
            correction = json.loads(response)
        entries.append(correction)
    # Store it this way so it's easy to skip if we have to resume
    corrections[subject] = entries
    if len(corrections) % 5 == 0:
        with open(args.output + ".wip", "w") as outfile:
            json.dump(corrections, outfile)
    
with open(args.output, "w") as outfile:
    final = []
    for subject in corrections:
        for entry in corrections[subject]:
            final.append(entry)
    json.dump(final, outfile)
