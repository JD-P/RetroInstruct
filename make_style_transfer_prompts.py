import os
import random
import json
import csv
import datasets
import datasets.distributed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

pg19 = datasets.load_dataset("pg19")

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
                                          use_fast=True)

index = {}
for i, title_author in enumerate(pg19["train"]["short_book_title"]):
    index[title_author] = i

with open("uncond_transfer_prompts.json") as infile:
    uncond_prompts = json.load(infile)

with open("cond_style_transfer_prompts.json") as infile:
    cond_prompts = json.load(infile)
    
with open("style_confabulations.json") as infile:
    bodies = json.load(infile)

def roll_style_marks():
    style_mark_pool = [{"start":"==START STYLE PASSAGE==","end":"==END STYLE PASSAGE=="},
                       {"start":"[BEGIN STYLE]","end":"[END STYLE]"},
                       {"start":"<STYLE>","end":"</STYLE>"},
                       {"start":"<BEGIN STYLE>","end":"<END STYLE>"},
                       {"start":"{{STYLE:START}}","end":"{{STYLE:END}}"},
                       {"start":"BEGIN STYLE]","end":"[END STYLE"},
                       {"start":"*STYLE START*","end":"*STYLE END*"},
                       {"start":"BEGIN STYLE TEXT","end":"CONCLUDE STYLE TEXT"},
                       {"start":"STYLE: START","end":"STYLE: END"},
                       {"start":"STYLE:","end":"END STYLE"},
                       {"start":"STYLE_START","end":"STYLE_END"},
                       {"start":"--START--","end":"--END--"},
                       {"start":"***START***","end":"***END***"},
                       {"start":"[STYLE:START]","end":"[STYLE:END]"},
                       {"start":"!BEGIN STYLE!","end":"!END STYLE!"},
                       {"start":"EXAMPLE PASSAGE","end":"END EXAMPLE"},
                       {"start":"EXAMPLE TEXT STYLE","end":"END EXAMPLE TEXT STYLE"},
                       {"start":"EXAMPLE_START","end":"EXAMPLE_END"},
                       {"start":"THE FOLLOWING PASSAGE","end":"END OF THE PREVIOUS PASSAGE"},
                       {"start":"BEGIN TARGET PASSAGE","end":"END TARGET PASSAGE"}]
    mark_roll = random.randrange(len(style_mark_pool))
    mismatch_roll = random.randrange(20)
    if mismatch_roll < 19:
        return mark_roll, style_mark_pool[mark_roll]["start"], style_mark_pool[mark_roll]["end"]
    else:
        corruption_roll = random.randrange(len(style_mark_pool))
        return (mark_roll,
                style_mark_pool[mark_roll]["start"],
                style_mark_pool[corruption_roll]["end"])

def roll_task_marks(style_mark_roll):
    task_mark_pool = [{"start":"==START TASK TEXT==","end":"==END TASK TEXT=="},
                      {"start":"[BEGIN TASK]","end":"[END TASK]"},
                      {"start":"<TASK>","end":"</TASK>"},
                      {"start":"<BEGIN TASK>","end":"<END TASK>"},
                      {"start":"{{TASK:START}}","end":"{{TASK:END}}"},
                      {"start":"TASK START]","end":"[END TASK"},
                      {"start":"*TASK START*","end":"*TASK END*"},
                      {"start":"BEGIN TASK TEXT","end":"CONCLUDE TASK TEXT"},
                      {"start":"TASK: START","end":"TASK: END"},
                      {"start":"TASK:","end":"END TASK"},
                      {"start":"TASK_START","end":"TASK_END"},
                      {"start":"--TASK--","end":"--END--"},
                      {"start":"***TASK***","end":"***END***"},
                      {"start":"[TASK:START]","end":"[TASK:END]"},
                      {"start":"!BEGIN TASK!","end":"!END TASK!"},
                      {"start":"REWRITE PASSAGE","end":"END OF REWRITE"},
                      {"start":"TASK TEXT","end":"END TASK TEXT"},
                      {"start":"TASK_START","end":"TASK_END"},
                      {"start":"THE TASK","end":"END OF THE TASK"},
                      {"start":"BEGIN REWRITE PASSAGE","end":"END REWRITE PASSAGE"}]
    mismatch_roll = random.randrange(20)
    # Match style marks
    if mismatch_roll < 18:
        return style_mark_roll, task_mark_pool[mark_roll]["start"], task_mark_pool[mark_roll]["end"]
    # Different task marks
    elif mismatch_roll == 18:
        corruption_roll = random.randrange(len(task_mark_pool))
        return (style_mark_roll,
                task_mark_pool[corruption_roll]["start"],
                task_mark_pool[corruption_roll]["end"])
    # Independent start and end task marks
    elif mismatch_roll == 19:
        corruption_roll_1 = random.randrange(len(task_mark_pool))
        corruption_roll_2 = random.randrange(len(task_mark_pool))
        return (style_mark_roll,
                task_mark_pool[corruption_roll_1]["start"],
                task_mark_pool[corruption_roll_2]["end"])

chartoks = 12000
context_third = 1300
title_authors = [ta for ta in bodies.keys()]
random.shuffle(title_authors)
if os.path.exists("style_passages.json"):
    with open("style_passages.json") as infile:
        episodes = json.load(infile)
else:
    episodes = []    
for i, title_author in enumerate(tqdm(title_authors)):
    if i % 1000 == 0:
        with open("style_passages.json", "w") as outfile:
            json.dump(episodes, outfile)
    try:
        assert pg19["train"]["short_book_title"][index[title_author]] == title_author
        book_text = pg19["train"]["text"][index[title_author]]
        if len(book_text) < chartoks:
            continue
            with open("log.txt", "a") as outfile:
                outfile.write(f"{title_author} was less than chartoks")
                outfile.flush()
    # Sometimes this does not result in a valid UTF-8 string -_-
    except:
        continue
        with open("log.txt", "a") as outfile:
            outfile.write(f"Couldn't process {title_author}, please investigate")
            outfile.flush()
    for i in range(5):
        if title_author in cond_prompts:
            prompt_open = cond_prompts[title_author][i]
        else:
            pick = random.randrange(len(uncond_prompts))
            prompt_open = uncond_prompts[pick][1]
        excerpt_start = random.randrange(len(book_text) - chartoks)
        passage = book_text[excerpt_start:excerpt_start+chartoks]
        passage = tokenizer.decode(
            tokenizer(passage, add_special_tokens=False)["input_ids"][:context_third],
        )
        mark_roll, start_style_mark, end_style_mark = roll_style_marks()
        _, start_task_mark, end_task_mark = roll_task_marks(mark_roll)
        
        episode = [title_author,
                   prompt_open,
                   start_style_mark,
                   passage,
                   end_style_mark,
                   start_task_mark,
                   bodies[title_author]["confabulations"][i],
                   end_task_mark,
                   bodies[title_author]["grounds"][i]]
        episodes.append(episode)
        # Wonky characters froze the terminal(???)
        #print(title_author,
        #      prompt_open,
        #      start_style_mark,
        #      passage[:256],
        #      "...",
        #      end_style_mark,
        #      start_task_mark,
        #      episode[6][:256],
        #      "...",
        #      end_task_mark,
        #      episode[8][:256],
        #      "...", sep="\n")

with open("retro_style_transfer.tsv", "w") as outfile:
    outwriter = csv.writer(outfile, dialect='excel-tab')
    outwriter.writerow(["title_author",
                        "prompt_open",
                        "start_style",
                        "style_passage",
                        "end_style",
                        "start_task",
                        "task_passage",
                        "end_task",
                        "ground_truth"])
    for episode in tqdm(episodes):
        outwriter.writerow(episode)
    outfile.flush()

