import codecs
import csv

# This list is not exhaustive and mostly exists to demonstrate word filtering
# and remind you to do it. Words are Rot13'd for politeness.
word_filter = {'ovgpu', 'phag', 'snttbgf', 'snttbg', 'snt', 'fuvg', 'cvff'}
with open("train.tsv", newline='') as infile:
    reader = csv.reader(infile, dialect="excel-tab")
    for row in reader:
        if codecs.encode(row[2], 'rot_13') in word_filter:
            continue
        # Note: I suggest formatting the dataset into multiple structures such as JSON
        # and XML in your dataloader. Future RetroInstruct modules will be designed to
        # more easily let you do this by prompting for structured data and then letting
        # the dataloader present that data in several formats. This will train
        # instruction models to fluidly work with both user prompts and structured
        # prompts more fitting for automated pipelines.
        # Concept Synthesis
        print("<s> [INST]"
              + row[0]
              + "\n\n"
              + row[3]
              + f"[/INST]1. {row[2].capitalize()} - "
              + row[4])
        # Word Parts
        print("<s> [INST]"
              + row[1].format(WORD=row[2])
              + "[/INST]"
              + row[3])
