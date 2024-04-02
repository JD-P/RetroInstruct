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

with open("rubric_seeds.json") as infile:
    rubric_seeds = json.load(infile)

with open("rubrics_prompt.txt") as infile:
    rubrics_prompt = infile.read()

rubrics = {}
for seed_set in tqdm(rubric_seeds):
    for seed in seed_set["questions"]:
        messages = [
            ChatMessage(role="user", content=rubrics_prompt.format(seed=seed))
        ]

        chat_response = client.chat(
            messages,
            model,
            response_format={"type": "json_object"},
        )

        rubric = json.loads(chat_response.choices[0].message.content)
        try:
            # Make sure we got output in expected format
            assert type(rubric) == dict
            len(rubric) == 1
            for question in rubric["rubric"]:
                assert type(question) == str
                assert not question.startswith(".")
                assert not question.startswith(">")
                assert not question.startswith(";")
                assert not question.startswith(":")
        except:
            # Give it another try
            print(rubric)
            print(f'"{seed}" failed. Retrying..."')
            rubric = json.loads(chat_response.choices[0].message.content)
        # Store it this way so it's easy to skip if we have to resume
        rubrics[seed] = rubric 
        if len(rubrics) % 100 == 0:
            with open("rubrics.json", "w") as outfile:
                json.dump(rubrics, outfile)
    
with open("rubrics.json", "w") as outfile:
    json.dump(rubrics, outfile)
