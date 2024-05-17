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
        self.analogies = datasets.load_dataset("jdpressman/retro-weave-eval-analogical-translations-v0.1")["train"]
        self.weave_eval_jdp = datasets.load_dataset("jdpressman/retro-weave-eval-jdp-v0.1")["train"]
        self.weave_eval_rubrics = datasets.load_dataset("jdpressman/retro-weave-eval-rubrics-v0.1")["train"]
        self.word_parts = datasets.load_dataset("jdpressman/retro-word-parts-v0.1")["train"]
        self.ascii_art = datasets.load_dataset("jdpressman/retro-ascii-art-v1")["train"]
        self.style_transfer = datasets.load_dataset("jdpressman/retro-text-style-transfer-v0.1")["train"]

        self.epoch = []
        self.epoch += self.prepare_analogies()
        self.epoch += self.prepare_jdp()
        self.epoch += self.prepare_ascii_art()
        self.epoch += self.prepare_word_parts()
        self.epoch += self.prepare_eval_rubrics()
        self.epoch += self.prepare_style_transfer()

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
                entries.append(mistral_instruction.format(question, response))
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
                entries.append(mistral_instruction.format(question, response))
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

            entries.append(mistral_instruction.format(instruction, output))
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
                out = f"<s> [INST]{ascii_art['prompt']}[/INST]"
            else:
                prefix = random.choice(self.art_prefixes)
                out = f"<s> [INST]{prefix} {ascii_art['prompt'].lower()}[/INST]"
            if mode == "aic":
                out += ascii_art['art_aic']
            elif mode == "i2a":
                out += ascii_art['art_i2a']
            entries.append(out)
        return entries

    def prepare_word_parts(self):
        word_filter = {'ovgpu', 'phag', 'snttbgf', 'snttbg', 'snt', 'fuvg', 'cvff'}
        entries = []
        for row in self.word_parts:
            if codecs.encode(row["word"], 'rot_13') in word_filter:
                continue
            # Note: I suggest formatting the dataset into multiple structures such as JSON
            # and XML in your dataloader. Future RetroInstruct modules will be designed to
            # more easily let you do this by prompting for structured data and then letting
            # the dataloader present that data in several formats. This will train
            # instruction models to fluidly work with both user prompts and structured
            # prompts more fitting for automated pipelines.
            # Concept Synthesis
            entries.append("<s> [INST]"
                           + row["synth_prompt"]
                           + "\n\n"
                           + row["part_list"]
                           + f"[/INST]1. {row['word'].capitalize()} - "
                           + row["word_guesses"] if row["word_guesses"] else "")
            # Word Parts
            entries.append("<s> [INST]"
                           + row["list_prompt"].format(WORD=row["word"])
                           + "[/INST]"
                           + row["part_list"])
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
            entries.append(mistral_instruction.format(prompt, output))
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
                    ''.join([row["prompt_open"],
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
                             row["end_task"],
                             "\n\n",
                             row["ground_truth"]])
                )
            # No seriously this dataset needs schema checking
            except TypeError:
                print("it happened again 2")
                continue
        return entries

dataloader = RetroInstructDataloader()

for entry in dataloader:
    print(entry)
