import random
import codecs
import json
import csv

with open("synth_prompts.json") as infile:
    synth_prompts = json.load(infile)

with open("list_prompts.json") as infile:
    list_prompts = json.load(infile)

with open("word_synthesis.json") as infile:
    word_synthesis = json.load(infile)

# Remove the ethnic slur nobody wants to train on
# You know the one. Rot13'd, Dead Dove: Do Not Eat
# Downstream dataloaders will have a more aggressive word filter
word_filter = {'avttre', 'avttref'}
words = [word for word in word_synthesis.keys()
         if codecs.encode(word, 'rot_13') not in word_filter]
random.shuffle(words)
with open("train.tsv", "w", newline='') as outfile:
    writer = csv.writer(outfile, dialect="excel-tab")
    writer.writerow(["synth_prompt",
                     "list_prompt",
                     "word",
                     "part_list",
                     "word_guesses"])
    for word in words[:-500]:
        synth_prompt = random.choice(synth_prompts)
        list_prompt = random.choice(list_prompts)
        writer.writerow([synth_prompt,
                         list_prompt,
                         word,
                         word_synthesis[word][0],
                         word_synthesis[word][1]])
        print(f"<s> [INST]{synth_prompt}\n\n{word_synthesis[word][0]}[/INST]1. {word.capitalize()} - "
              + word_synthesis[word][1])

with open("val.tsv", "w", newline='') as outfile:
    writer = csv.writer(outfile, dialect="excel-tab")
    writer.writerow(["synth_prompt",
                     "list_prompt",
                     "word",
                     "part_list",
                     "word_guesses"])
    for word in words[-500:]:
        synth_prompt = random.choice(synth_prompts)
        list_prompt = random.choice(list_prompts)
        writer.writerow([synth_prompt,
                         list_prompt,
                         word,
                         word_synthesis[word][0],
                         word_synthesis[word][1]])
        print(f"<s> [INST]"
              + list_prompt.format(WORD=word)
              + "[/INST]"
              + word_synthesis[word][0])
