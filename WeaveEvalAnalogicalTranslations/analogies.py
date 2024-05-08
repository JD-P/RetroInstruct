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
parser.add_argument("-o", "--output", default="analogies.json")
parser.add_argument("-r", "--resume")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

with open("128_controversies.txt") as infile:
    subjects = [subject.lower().strip() for subject in infile.readlines()]
    
with open(args.prompt) as infile:
    prompt = infile.read()

if args.resume:
    with open(args.output) as infile:
        analogies = json.load(infile)
else:
    analogies = {}

for subject in tqdm(subjects):
    if subject in analogies:
        continue
    messages = [
        # ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prompt.replace("INSERT_SUBJECT_HERE", subject))
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
        analogy_batch = json.loads(response)
        assert type(analogy_batch) == dict
        len(analogy_batch["entries"]) == 6
        for analogy in analogy_batch["entries"]:
            assert "subject" in analogy
            assert type(analogy["subject"]) == str
            assert "position" in analogy
            assert analogy["position"].lower() in ["for", "against"]
            assert "salient-features" in analogy
            assert type(analogy["salient-features"]) == list
            assert "reference-class" in analogy
            assert type(analogy["reference-class"]) == str
            assert "prior-arguments" in analogy
            assert type(analogy["prior-arguments"]) == list
            assert "chosen-argument" in analogy
            assert type(analogy["chosen-argument"]) == str
            assert "differences" in analogy
            assert type(analogy["differences"]) == list
            assert "analogical-translation" in analogy
            assert type(analogy["analogical-translation"]) == str
            assert "corruptions" in analogy
            assert type(analogy["corruptions"]) == list
    except:
        # Give it another try
        print(analogy_batch)
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
        analogy_batch = json.loads(response)
    # Store it this way so it's easy to skip if we have to resume
    analogies[subject] = analogy_batch["entries"]
    if len(analogies) % 10 == 0:
        with open(args.output, "w") as outfile:
            json.dump(analogies, outfile)
    
with open(args.output, "w") as outfile:
    json.dump(analogies, outfile)
