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
from transformers.generation.streamers import BaseStreamer
import datasets
import datasets.distributed
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from lora_tune import lora_tune_evaluator
from dataset import ZippedConversationsDataset

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

class ProgressBarStreamer(BaseStreamer):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs.setdefault("unit", "tok")
        self.next_tokens_are_prompt = True
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()

    def put(self, value):
        if not self.next_tokens_are_prompt:
            self.pbar.update(value.numel())
        self.next_tokens_are_prompt = False

    def end(self):
        self.next_tokens_are_prompt = True
        
@torch.no_grad()
def generate_outputs(generator, text, n_tokens, n=1, batch_size=1):
    tokenizer, model = generator

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096 - n_tokens,
    ).to("cuda")

    outputs = []
    with ProgressBarStreamer(total=n_tokens * n) as pbar:
        for i in range(0, n, batch_size):
            n_batch = min(batch_size, n - i)
            input_ids = inputs.input_ids.tile((n_batch, 1))
            attention_mask = inputs.attention_mask.tile((n_batch, 1))
            outputs_batch = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1,
                top_k=50,
                repetition_penalty=1.02,
                min_new_tokens=16,
                max_new_tokens=n_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=13,
                streamer=pbar,
            )
            outputs.append(outputs_batch)

    outputs = torch.cat(outputs)
    out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in outputs]
    in_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    return [out_text[in_length:] for out_text in out_texts]
        
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

def prepare_rubric(rubric_path, evaluator):
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
    for principle in rubric["principles"]:
        evaluation_prompt = principle["body"].format(preamble=rubric["preamble"])
        score_prompt_fn = partial(make_score_prompt_fn, evaluator)
        score_prompt_fn = partial(score_prompt_fn, evaluation_prompt)
        # FLAN evaluator LoRA suffix
        rubric_score_fns.append(partial(score_prompt_fn, "<|end|>"))
    return rubric_score_fns, principle_weights, principle_signs

def gen_epoch(x, n):
    combos = []
    combo = [1 for i in range(x)]
    combos.append(combo.copy())
    while combo[-1] < n+1:
        combo[0] += 1
        for i in range(x):
            try:
                if combo[i] > n:
                    combo[i+1] += 1
                    combo[i] = 1
            except IndexError:
                
                return combos
        combos.append(combo.copy())
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generator", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--evaluator", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    args = parser.parse_args()

    device = "cuda:0"
    tokenizer, model = load_generator_evaluator(args.generator, args.evaluator)
    evaluator = generator = (tokenizer, model)
    # adapter_name = "generator" if "generator" in generator[1].peft_config else None
    generate_fn = partial(generate_outputs, generator, n=5, batch_size=5)
    evaluate_fn = partial(evaluate_outputs, evaluator)

    pg19 = datasets.load_dataset("pg19")
    with open("prompts/cond_style_transfer_opening_rewrites.txt") as infile:
        style_transfer_prompt_template = infile.read()
    # TODO: Change weave to let me use q_weights and q_signs
    rubric_score_fns, q_weights, q_signs = prepare_rubric("prompts/uncond_style_transfer_prompt_eval.txt",
                                                          evaluator)
    openings = {}
    shuffled_titles = pg19["train"]["short_book_title"].copy()
    random.shuffle(shuffled_titles)
    for i in tqdm(range(5000)):
        title_author = shuffled_titles[i]
        
        prompt = (style_transfer_prompt_template
                  + '{"title-author": "' + title_author + '", "prompt": "')
        openings[title_author] = [out[:-3] for out in generate_fn(prompt, 128)]
#            import pdb
#            pdb.set_trace()
#            retries = 0
#            while not out_prompt.endswith('"}\n'):
#                out_prompt = generate_fn(prompt, 128)[0]
#                retries += 1
#                if retries > 10:
#                    # These can be detected later as duds
#                    out_prompt = ''
#                    break
#            openings[title_author].append(out_prompt[:-3])
        for j in range(5):
            print(openings[title_author][j])
    with open("cond_style_transfer_prompts.json", "w") as outfile:
        json.dump(openings, outfile)
