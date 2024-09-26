from argparse import ArgumentParser
import os
import re
import json
import time
import datetime
import random
import hashlib
import zipfile
from contextlib import contextmanager
from functools import partial
from itertools import islice
from tqdm import tqdm
import torch
from weave import generate_outputs_vllm, evaluate_outputs_vllm
from weave import make_score_prompt_vllm

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def parse_constitution(cons):
    principles = {}
    raw_principles = re.split("==\[(.+)\]==", cons)[1:]
    principle_pairs = [i for i in batched(raw_principles, 2)]
    principle_pairs = [(i[0].strip(), i[1].strip()) for i in principle_pairs]
    principles["preamble"] = principle_pairs[0][1]
    principles["principles"] = []
    for pair in principle_pairs[1:]:
        principle = {}
        for parameter in pair[0].split(";"):
            try:
                name, value = parameter.split(":")
            except ValueError:
                raise ValueError(f"{pair} is missing a colon in a header value")
            principle[name.strip().lower()] = value.strip().lower()
        principle["body"] = pair[1].strip()
        principles["principles"].append(principle)
    return principles

def prepare_rubric(rubric_path, rubric_score_fn, prompt):
    with open(rubric_path) as infile:
        rubric = parse_constitution(infile.read())
        principle_weights = [float(principle["weight"]) for principle in rubric["principles"]]
        principle_weights = torch.tensor(principle_weights)
        principle_signs = []
        for principle in rubric["principles"]:
            answer = principle["answer"].lower()
            if answer not in {"yes", "no"}:
                raise ValueError("desired answer must be yes or no")
            principle_signs.append(1 if answer == "yes" else -1)
        principle_signs = torch.tensor(principle_signs)
    rubric_score_fns = []
    for principle in rubric["principles"]:
        evaluation_prompt = principle["body"].format(preamble=rubric["preamble"],
                                                     prompt=prompt,
                                                     response="{response}")
        score_prompt_fn = partial(rubric_score_fn, evaluation_prompt)
        # FLAN evaluator LoRA suffix
        rubric_score_fns.append(partial(score_prompt_fn, "<|end|>", prompt))
    return rubric_score_fns, principle_weights, principle_signs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("rubric_path", help="Filepath to the grading rubric to use.")
    parser.add_argument("cases", help="Filepath to the test cases to evaluate.")
    parser.add_argument("--evaluator", default="upstage/SOLAR-10.7B-v1.0")
    args = parser.parse_args()

    evaluate_fn = partial(evaluate_outputs_vllm, args.evaluator)

    rubric_score_fns, q_weights, q_signs = prepare_rubric(args.rubric_path,
                                                          make_score_prompt_vllm,
                                                          "")
        
    with open(args.cases) as infile:
        cases = infile.read().split("<|endcase|>")

    # Evaluate each market detail using the evaluate_outputs_vllm function
    for case in tqdm(cases, desc="Evaluating cases"):
        # Evaluate the input text using the evaluate_outputs_vllm function
        score = evaluate_fn(rubric_score_fns, [case], port=5001)[0].item()
        print(score, case[:150] + "...", end="\n\n")
