import random
import csv

rows = []
with open("retro_style_transfer.tsv") as infile:
    reader = iter(csv.reader(infile, dialect="excel-tab"))
    has_next = True
    while has_next:
        try:
            row = next(reader)
            rows.append(row)
        except csv.Error:
            continue
        except StopIteration:
            has_next = False

random.shuffle(rows)
val_set_size = round(len(rows) / 100) * 3
train = rows[1:-val_set_size]
val = rows[-val_set_size:]

with open("train.tsv", "w") as outfile:
    writer = csv.writer(outfile, dialect="excel-tab")
    writer.writerow(["title_author",
                     "prompt_open",
                     "start_style",
                     "style_passage",
                     "end_style",
                     "start_task",
                     "task_passage",
                     "end_task",
                     "ground_truth"])
    for row in train:
        writer.writerow(row)
    outfile.flush()

with open("val.tsv", "w") as outfile:
    writer = csv.writer(outfile, dialect="excel-tab")
    writer.writerow(["title_author",
                     "prompt_open",
                     "start_style",
                     "style_passage",
                     "end_style",
                     "start_task",
                     "task_passage",
                     "end_task",
                     "ground_truth"])
    for row in val:
        writer.writerow(row)
    outfile.flush()
