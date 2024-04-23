import json
import random
from tqdm import tqdm
from argparse import ArgumentParser
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

parser = ArgumentParser()
parser.add_argument("passages", nargs="+")
parser.add_argument("--api-key")
parser.add_argument("--prompt")
parser.add_argument("--temp", default=1, type=float)
parser.add_argument("-o", "--output", default="passages.json")
parser.add_argument("-r", "--resume")
args = parser.parse_args()

with open(args.api_key) as infile:
    api_key = infile.read().strip()
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

passages = []
for corpus_path in args.passages:
    with open(corpus_path) as infile:
        corpus = json.load(infile)
        for entry in corpus:
            passages.append(corpus[entry]["passage"])

with open(args.prompt) as infile:
    prompt = infile.read()
    
with open("quiz_system_prompt.txt") as infile:
    system_prompt = infile.read()

if args.resume:
    with open(args.output) as infile:
        questions = json.load(infile)
else:
    questions = {}

for passage in tqdm(passages):
    if passage in questions:
        continue
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=prompt.format(passage=passage))
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
        question_set = json.loads(response)
        assert type(question_set) == list
        len(question_set) == 5
        for question in question_set:
            assert type(question) == dict
            assert "type" in question
            assert "question" in question
            assert "label" in question
            assert "explanation" in question
            assert question["label"].lower() in {"yes", "no"}
            assert type(question["question"]) == str
            assert type(question["explanation"]) == str
    except:
        # Give it another try
        print(question_set)
        print(f'"{passage}" failed. Retrying..."')
        chat_response = client.chat(
            messages,
            model,
            temperature=args.temp,
            response_format={"type": "json_object"},
        )
        response = chat_response.choices[0].message.content
        response = response.replace("\\n", "\n")
        response = response.replace("\n", "\\n")
        question_set = json.loads(response)
    # Store it this way so it's easy to skip if we have to resume
    questions[passage] = question_set 
    if len(questions) % 10 == 0:
        with open(args.output, "w") as outfile:
            json.dump(questions, outfile)
    
with open(args.output, "w") as outfile:
    json.dump(questions, outfile)
