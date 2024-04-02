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

with open("rubric_themes.txt") as infile:
    rubric_themes = infile.readlines()

with open("rubric_seed_prompt.txt") as infile:
    rubric_seed_prompt = infile.read()

seeds = []
for rubric_theme in tqdm(rubric_themes):
    messages = [
        ChatMessage(role="user", content=rubric_seed_prompt.format(theme=rubric_theme))
    ]

    chat_response = client.chat(
        messages,
        model,
        response_format={"type": "json_object"},
    )

    seeds.append(json.loads(chat_response.choices[0].message.content))
    if len(seeds) % 50 == 0:
        with open("rubric_seeds.json", "w") as outfile:
            json.dump(seeds, outfile)
    
with open("rubric_seeds.json", "w") as outfile:
    json.dump(seeds, outfile)
