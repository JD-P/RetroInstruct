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

def prepare_rubric(rubric_path, rubric_score_fn):
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
                                                     text="{text}")
        score_prompt_fn = partial(rubric_score_fn, evaluation_prompt)
        # FLAN evaluator LoRA suffix
        rubric_score_fns.append(partial(score_prompt_fn, "<|end|>"))
    return rubric_score_fns, principle_weights, principle_signs

def format_market_details(market):
    question = market.get("question")
    yes_probability = market.get("probability") * 100
    no_probability = (1 - market.get("probability")) * 100
    unique_bettor_count = market.get("uniqueBettorCount")
    creator_name = market.get("creatorName")
    created_time = datetime.datetime.fromtimestamp(market.get("createdTime") / 1000).strftime("%Y-%m-%d at %H:%M UTC")
    close_time = datetime.datetime.fromtimestamp(market.get("closeTime") / 1000).strftime("%Y-%m-%d at %H:%M UTC")
    text_description = market.get("textDescription")
    resolution = market.get("resolution").title() + "."
    out = ""
    out += "Manifold Markets\n\n"
    out += f"{question}\n"
    out += f"YES {yes_probability:.2f}% NO {no_probability:.2f}% "
    out += f"| {unique_bettor_count} Bettors\n"
    out += f"Creator: {creator_name}\n"
    out += f"Created: {created_time}\n"
    out += f"Closes: {close_time}\n\n"
    out += f"Description & Resolution Criteria: {text_description}\n\n"
    out += f"Resolution: {resolution}"
    return out

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("rubric_paths", nargs="+",
                        help="Filepaths to the grading rubrics to use.")
    parser.add_argument("--market-details-path", help="Filepath to the formatted market details to evaluate.")
    parser.add_argument("--evaluator", default="upstage/SOLAR-10.7B-v1.0")
    args = parser.parse_args()

    evaluate_fn = partial(evaluate_outputs_vllm, args.evaluator)

    rubrics = {}
    for rubric_path in args.rubric_paths:
        rubric_name = os.path.split(rubric_path)[-1].split("_")[0]
        rubric_score_fns, q_weights, q_signs = prepare_rubric(rubric_path,
                                                              make_score_prompt_vllm)
        rubrics[rubric_name] = rubric_score_fns
        
    with open(args.market_details_path) as infile:
        market_details = json.load(infile)

    if os.path.exists("market_detail_scores_2024_06_05.json"):
        with open("market_detail_scores_2024_06_05.json") as infile:
            market_scores = json.load(infile)
    else:
        market_scores = {}
    # Evaluate each market detail using the evaluate_outputs_vllm function
    for market_detail in tqdm(market_details, desc="Evaluating market details"):
        if market_detail["id"] in market_scores:
            continue
        if (not market_detail.get("isResolved") or
            market_detail.get("outcomeType") != "BINARY"):
            continue
        if market_detail.get("resolution") not in {"YES", "NO"}:
            continue

        # Concatenate the question and text description to form the input text
        input_text = format_market_details(market_detail)

        # Evaluate the input text using the evaluate_outputs_vllm function
        try:
            resolvable_score = evaluate_fn(rubrics["resolvable"], [input_text])[0].item()
            personal_score = evaluate_fn(rubrics["personal"], [input_text])[0].item()
            degeneracy_score = evaluate_fn(rubrics["degeneracy"], [input_text])[0].item()
        except:
            input("Check server up before pressing enter to continue:")
            continue
        market_scores[market_detail["id"]] = {"resolvable":resolvable_score,
                                              "personal":personal_score,
                                              "degeneracy": degeneracy_score}
        if len(market_scores) % 500 == 0:
            with open("market_detail_scores_2024_06_05.json", "w") as outfile:
                json.dump(market_scores, outfile)
    with open("market_detail_scores_2024_06_05.json", "w") as outfile:
        json.dump(market_scores, outfile)
