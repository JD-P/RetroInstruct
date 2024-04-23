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
parser.add_argument("-o", "--output", default="passages.json")
parser.add_argument("-r", "--resume")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

with open("themes.txt") as infile:
    themes = [theme.lower().strip() for theme in infile.readlines()]

with open("start_words.txt") as infile:
    start_words = [word.strip() for word in infile.readlines()]
    
with open(args.prompt) as infile:
    prompt = infile.read()
    
with open("chatjdp_system_prompt.txt") as infile:
    system_prompt = infile.read()

if args.resume:
    with open(args.output) as infile:
        passages = json.load(infile)
else:
    passages = {}

for theme in tqdm(themes):
    if theme in passages:
        continue
    start_word = random.choice(start_words)
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prompt.format(theme=theme, start_word=start_word))
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
        passage = json.loads(response)
        assert type(passage) == dict
        len(passage) == 1
        assert type(passage["passage"]) == str
        assert passage["passage"].startswith(start_word)
    except:
        # Give it another try
        print(passage)
        print(f'"{theme}" failed. Retrying..."')
        chat_response = client.chat(
            messages,
            model,
            temperature=args.temp,
            response_format={"type": "json_object"},
        )
        response = chat_response.choices[0].message.content
        response = response.replace("\\n", "\n")
        response = response.replace("\n", "\\n")
        passage = json.loads(response)
    # Store it this way so it's easy to skip if we have to resume
    passages[theme] = passage 
    if len(passages) % 10 == 0:
        with open(args.output, "w") as outfile:
            json.dump(passages, outfile)
    
with open(args.output, "w") as outfile:
    json.dump(passages, outfile)
