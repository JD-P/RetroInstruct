import random
import codecs
import json
import datasets

class RetroInstructDataloader:
    def __init__(self):
        # Load yes or no prefix strings
        with open("yes_or_no.txt", "r") as file:
            self.yes_no_prefixes = [line.strip() for line in file.readlines()]
        with open("art_prefixes.txt") as infile:
            self.art_prefixes = [line.strip() for line in infile.readlines()]
        self.analogies = (
            [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-analogical-translations-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-analogical-translations-v0.1")["validation"]]
        )
        random.shuffle(self.analogies)
        self.weave_eval_jdp = (
            [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-jdp-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-jdp-v0.1")["validation"]]
        )
        random.shuffle(self.weave_eval_jdp)
        self.weave_eval_rubrics = (
            [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-rubrics-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-weave-eval-rubrics-v0.1")["validation"]]
        )
        random.shuffle(self.weave_eval_rubrics)
        self.word_parts = (
            [i for i in datasets.load_dataset("jdpressman/retro-word-parts-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-word-parts-v0.1")["validation"]]
        )
        random.shuffle(self.word_parts)
        self.ascii_art = (
            [i for i in datasets.load_dataset("jdpressman/retro-ascii-art-v1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-ascii-art-v1")["validation"]]
        )
        random.shuffle(self.ascii_art)
        self.style_transfer = (
            [i for i in datasets.load_dataset("jdpressman/retro-text-style-transfer-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-text-style-transfer-v0.1")["validation"]]
        )
        random.shuffle(self.style_transfer)
        self.easy_prose_diffs = (
            [i for i in datasets.load_dataset("jdpressman/retro-easy-prose-repair-diffs-v0.1")["train"]]
            + [i for i in datasets.load_dataset("jdpressman/retro-easy-prose-repair-diffs-v0.1")["validation"]]
        )
        random.shuffle(self.easy_prose_diffs)
        
        self.epoch = []
        self.epoch += self.prepare_analogies()
        self.epoch += self.prepare_jdp()
        self.epoch += self.prepare_ascii_art()
        self.epoch += self.prepare_word_parts()
        self.epoch += self.prepare_eval_rubrics()
        self.epoch += self.prepare_style_transfer()
        self.epoch += self.prepare_easy_prose_diffs()

        random.shuffle(self.epoch)

    def __iter__(self):
        return iter(self.epoch)
        
    def prepare_analogies(self):
        good_faith_argument_questions = [
            "Is the following passage a good faith argument?",
            "Is this a good faith argument?",
            "Do you think this argument is in good faith?",
            "Does the following text seem like an attempt at truthseeking?",
            "Would you say this author is honest, nuanced, truthful, etc?",
            "Does this author seem like they could pass an ideological turing test?"]
    
        mistral_instruction = "<s> [INST]{}[/INST]{}"
        entries = []
        for entry in self.analogies:
            footnotes = "\n".join([f"[{i}]: {footnote}" for i, footnote in enumerate(entry["critique"])])
            features = "\n".join(entry["salient-features"])
            priors = "\n".join(entry["prior-arguments"])
            differences = "\n".join(entry["differences"])
            yes_no_prefix = random.choice(self.yes_no_prefixes)
            good_faith = random.choice(good_faith_argument_questions)

            if random.randrange(2):
                question = (yes_no_prefix
                            + " " + good_faith
                            + "\n\n" + entry["analogical-translation"])
                response = (entry["label"] + ".\n\n"
                            + "Criticism:\n" + footnotes + "\n\n"
                            + "Subject: " + entry["subject"] + "\n\n"
                            + "Position: " + entry["position"] + "\n\n"
                            + "Salient Features:\n" + features + "\n\n"
                            + "Reference Class: " + entry["reference-class"] + "\n\n"
                            + "Prior Arguments:\n" + priors + "\n\n"
                            + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                            + "Differences:\n" + differences + "\n\n")
                entries.append({"inputs":question, "targets":response})
            else:
                question = (entry["analogical-translation"] + "\n\n"
                            + "Criticism:\n" + footnotes + "\n\n"
                            + "Subject: " + entry["subject"] + "\n\n"
                            + "Position: " + entry["position"] + "\n\n"
                            + "Salient Features:\n" + features + "\n\n"
                            + "Reference Class: " + entry["reference-class"] + "\n\n"
                            + "Prior Arguments:\n" + priors + "\n\n"
                            + "Chosen Argument: " + entry["chosen-argument"] + "\n\n"
                            + "Differences:\n" + differences + "\n\n"
                            + f"{good_faith}")
                response = entry["label"]
                entries.append({"inputs":question, "targets":response})
        return entries

    def prepare_jdp(self):
        mistral_instruction = "<s> [INST]{}[/INST]{}"
        entries = []
        for row in self.weave_eval_jdp:
            passage = row["passage"]
            questions = row["questions"]
            random.shuffle(questions)

            # Choose a random yes/no prefix
            yes_no_prefix = random.choice(self.yes_no_prefixes)

            instruction = f"{yes_no_prefix} In the following passage:\n\n<passage>\n{passage}\n</passage>\n\n"
            output = ""

            for i, question in enumerate(questions):
                question_text = question["question"]
                label = question["label"]
                explanation = question["explanation"]

                if i < len(questions) - 1:
                    instruction += f"{question_text} {label}. {explanation}\n\n"
                else:
                    instruction += f"{question_text}"
                    output += f"{label}. {explanation}"

            entries.append({"inputs":instruction, "targets":output})
        return entries
    
    def prepare_ascii_art(self):
        indices = [i for i in range(len(self.ascii_art))]
        epoch = []
        for i in indices:
            epoch.append((self.ascii_art[i], "aic"))
            epoch.append((self.ascii_art[i], "i2a"))

        random.shuffle(epoch)
        entries = []
        for ascii_art, mode in epoch:
            roll = random.randrange(5)
            if roll == 0:
                inputs = ascii_art['prompt']
            else:
                prefix = random.choice(self.art_prefixes)
                inputs = f"{prefix} {ascii_art['prompt'].lower()}"
            if mode == "aic":
                art = ascii_art['art_aic']
            elif mode == "i2a":
                art = ascii_art['art_i2a']
            entries.append({"inputs":inputs, "targets":art})
        return entries

    def prepare_word_parts(self):
        word_filter = {'ovgpu', 'phag', 'snttbgf', 'snttbg', 'snt', 'fuvg', 'cvff'}
        entries = []
        # These are overrepresented if we don't cap how many we use
        for row in self.word_parts[:5000]:
            if codecs.encode(row["word"], 'rot_13') in word_filter:
                continue
            # Note: I suggest formatting the dataset into multiple structures such as JSON
            # and XML in your dataloader. Future RetroInstruct modules will be designed to
            # more easily let you do this by prompting for structured data and then letting
            # the dataloader present that data in several formats. This will train
            # instruction models to fluidly work with both user prompts and structured
            # prompts more fitting for automated pipelines.
            # Concept Synthesis
            entries.append(
                {"inputs":(row["synth_prompt"]
                           + "\n\n"
                           + row["part_list"]),
                 "targets":(f"1. {row['word'].capitalize()} - "
                            + row["word_guesses"] if row["word_guesses"] else "")}
            )
            # Word Parts
            entries.append({"inputs":row["list_prompt"].format(WORD=row["word"]),
                            "targets":row["part_list"]})
        return entries

    def prepare_eval_rubrics(self):
        mistral_instruction = "<s> [INST]{}[/INST]{}"
        entries = []
        for row in self.weave_eval_rubrics:
            prompt = row["prompt_open"].format(seed=row["seed"])
            output = ""
            index = 1
            for item in row["rubric"]:
                output += (str(index) + ". " + item + "\n")
                index += 1
            entries.append({"inputs":prompt, "targets":output})
        return entries

    def prepare_style_transfer(self):
        entries = []
        # I made too many of these
        for i, row in enumerate(self.style_transfer):
            if i > 5000:
                break
            # TODO: Update style transfer set with proper schema
            if type(row) != dict:
                print("it happened again")
                continue
            try:
                entries.append(
                    {"inputs":''.join([row["prompt_open"],
                                       "\n\n",
                                       row["start_style"],
                                       "\n",
                                       row["style_passage"],
                                       "\n",
                                       row["end_style"],
                                       "\n\n",
                                       row["start_task"],
                                       "\n",
                                       row["task_passage"],
                                       "\n",
                                       row["end_task"]]),
                     "targets":row["ground_truth"]}
                )
            # No seriously this dataset needs schema checking
            except TypeError:
                print("it happened again 2")
                continue
        return entries

    def prepare_easy_prose_diffs(self):
        mistral_instruction = "<s> [INST]{}[/INST]{}"
        entries = []
        diff_formats = ["gnudiff", "git", "dmp"]

        for diff in self.easy_prose_diffs:
            diff_format = random.choice(diff_formats)
            if diff_format == "gnudiff":
                instruction = diff["gnudiff_instruction"]
                diff_text = diff["gnudiff"]
            elif diff_format == "git":
                instruction = diff["gitdiff_instruction"]
                diff_text = diff["gitdiff"]
            elif diff_format == "dmp":
                instruction = diff["dmpdiff_instruction"]
                diff_text = diff["dmpdiff"]
            entries.append(
                {"inputs":''.join([
                    instruction,
                    "\n\n",
                    "<passage>\n",
                    diff["text_corrupted"],
                    "\n</passage>",
                    "<|end|>"]),
                 "targets":''.join([
                     "<diagnosis>\n",
                     diff["operations"],
                     "</diagnosis>\n",
                     "<diff>\n",
                     diff_text,
                     "</diff>\n",
                     "<repaired>\n",
                     diff["text_clean"],
                     "</repaired>"])
                }
            )
        return entries

    
dataloader = RetroInstructDataloader()

dataset = [entry for entry in dataloader]

val_start = len(dataset) - round(len(dataset) / 10)

with open("train.json", "w") as outfile:
    json.dump(dataset[:val_start], outfile)

with open("val.json", "w") as outfile:
    json.dump(dataset[val_start:], outfile)
    
for entry in dataloader:
    assert type(entry["inputs"]) == str
    assert type(entry["targets"]) == str
    print(entry)
