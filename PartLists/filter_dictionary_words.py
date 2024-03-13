from argparse import ArgumentParser
import os
import re
import json
import time
import random
import hashlib
import zipfile
from contextlib import contextmanager
from functools import partial
from itertools import islice
from flask import Flask, request, jsonify, make_response
from tqdm import tqdm
import torch
import torch.nn as nn
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from lora_tune import lora_tune_evaluator

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)

def load_generator_evaluator(generator_adapter_name, evaluator_adapter_name):
    # peft_config = peft.PeftConfig.from_pretrained(evaluator_adapter_name)
    # model_name = peft_config.base_model_name_or_path
    model_name = evaluator_adapter_name
    tokenizer = AutoTokenizer.from_pretrained(evaluator_adapter_name)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True,
    )
    # model = peft.PeftModel.from_pretrained(model, evaluator_adapter_name, "evaluator")
    # if generator_adapter_name:
    #    model.load_adapter(generator_adapter_name, "generator")
    # peft_config = peft.LoraConfig(
    #    peft.TaskType.CAUSAL_LM,
    #    inference_mode=True,
    #    r=32,
    #    lora_alpha=8,
    #    lora_dropout=0.0,
    #    target_modules=[
    #        "self_attn.q_proj",
    #        "self_attn.k_proj",
    #        "self_attn.v_proj",
    #        "self_attn.o_proj",
    #        "mlp.gate_proj",
    #        "mlp.up_proj",
    #        "mlp.down_proj",
    #    ],
    # )
    return tokenizer, model

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def make_score_prompt_fn(template, suffix, word, _placeholder):
    return template.format(WORD="{WORD}", FILTER_WORD=word) + suffix
        
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

def prepare_rubric(rubric_path):
    with open(rubric_path) as infile:
        rubric = parse_constitution(infile.read())
        principle_weights = [float(principle["weight"]) for principle in rubric["principles"]]
        principle_weights = torch.tensor(principle_weights, device=device)
        principle_signs = []
        for principle in rubric["principles"]:
            answer = principle["answer"].lower()
            if answer not in {"yes", "no"}:
                raise ValueError("desired answer must be yes or no")
            principle_signs.append(1 if answer == "yes" else -1)
        principle_signs = torch.tensor(principle_signs, device=device)
    rubric_score_fns = []
    principle = rubric["principles"][0]
    evaluation_prompt = principle["body"].format(preamble=rubric["preamble"],
                                                 WORD="{WORD}",
                                                 FILTER_WORD="{FILTER_WORD}")
    return evaluation_prompt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dictionary")
    parser.add_argument("--generator", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--evaluator", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--batch-size", default=32)
    args = parser.parse_args()

    device = "cuda:0"
    tokenizer, model = load_generator_evaluator(args.generator, args.evaluator)
    evaluator = generator = (tokenizer, model)
    # adapter_name = "generator" if "generator" in generator[1].peft_config else None
    generate_fn = partial(generate_outputs, generator, batch_size=1)
    evaluate_fn = partial(evaluate_outputs, evaluator)

    # Prefilter dictionary
    # Debian dictionary path
    with open(args.dictionary) as infile:
        words = [i.strip() for i in infile.readlines() if not i.endswith("'s\n")]
    valid_words = []
    # Pool so TQDM can give estimated completion time
    word_batches = [i for i in batched(words, args.batch_size)]
    for word_batch in tqdm(word_batches):
        evaluation_prompt = prepare_rubric("prompts/filter_dictionary_rubric.txt")
        score_prompt_fn = partial(make_score_prompt_fn, evaluation_prompt)
        # Mixtral Instruct suffix
        score_prompt_fns = [partial(score_prompt_fn, "[/INST]"),]
        # Change name to avoid overwriting global baseline evaluate_fn partial
        score_fn = partial(evaluate_fn, score_prompt_fns)
        scores = score_fn([[word, ''] for word in word_batch])
        for i, score in enumerate(scores):
            if score.item() > 10:
                valid_words.append(word_batch[i])
                print(f"{word_batch[i]} accepted")
    with open("valid_part_words.json", "w") as outfile:
        json.dump(valid_words, outfile)
