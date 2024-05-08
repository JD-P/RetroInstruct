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
parser.add_argument("-o", "--output", default="prompts.json")
parser.add_argument("-r", "--resume")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

with open("subjects.txt") as infile:
    subjects = [subject.lower().strip() for subject in infile.readlines()]

with open("styles.txt") as infile:
    styles = [style.lower().strip() for style in infile.readlines()]
    
with open(args.prompt) as infile:
    user_prompt = infile.read()

if args.resume:
    with open(args.resume) as infile:
        prompts = json.load(infile)
else:
    prompts = {}

for subject in tqdm(subjects):
    if subject in prompts:
        continue
    prompts[subject] = []
    batch_styles = random.sample(styles, 10)
    for style in batch_styles:
        filled_prompt = user_prompt.replace("REPLACE_WITH_SUBJECT", subject)
        filled_prompt = filled_prompt.replace("REPLACE_WITH_STYLE", style)
        messages = [
            # ChatMessage(role="system", content=system_prompt),

            ChatMessage(role="user", content=filled_prompt)
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
            prompt_batch = json.loads(response)
            assert type(prompt_batch) == dict
            len(prompt_batch["prompts"]) == 5
            for prompt in prompt_batch["prompts"]:
                assert "subject" in prompt
                assert "style" in prompt
                assert "prompt" in prompt
        except:
            # Give it another try
            print(prompt_batch)
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
            prompt_batch = json.loads(response)
        # Store it this way so it's easy to skip if we have to resume
        prompts[subject] += prompt_batch["prompts"]
    if len(prompts) % 5 == 0:
        with open(args.output + ".wip", "w") as outfile:
            json.dump(prompts, outfile)
    
with open(args.output, "w") as outfile:
    final = []
    for subject in prompts:
        for prompt in prompts[subject]:
            final.append(prompt)
    json.dump(final, outfile)
