from argparse import ArgumentParser
import os
import re
import json
import time
import random
import hashlib
import zipfile
import asyncio
from contextlib import contextmanager
from functools import partial
from itertools import islice
from tqdm import tqdm
import requests
import accelerate
import torch
import torch.nn as nn
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from weave import weave_tree_search, generate_outputs, evaluate_outputs
from weave import make_score_prompt_fn, TreeNode
from transformers.generation.streamers import BaseStreamer


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

def make_score_prompt_fn(template, suffix, word, part_list):
    return template.format(prompt=word, response=part_list) + suffix
        
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
                                                 prompt="{prompt}",
                                                 response="{response}")
    return evaluation_prompt


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
                min_new_tokens=n_tokens,
                max_new_tokens=n_tokens,
                pad_token_id=tokenizer.eos_token_id,
                streamer=pbar,
            )
            outputs.append(outputs_batch)

    outputs = torch.cat(outputs)
    outputs = accelerator.gather(outputs)
    out_texts = [tokenizer.decode(toks, skip_special_tokens=True) for toks in outputs]
    in_lengths = [len(tokenizer.decode(toks, skip_special_tokens=True))
                  for toks in inputs.input_ids]
    return [out_texts[i][in_lengths[i]:] for i in range(len(out_texts))]

async def generate_from_vllm(model_name, prompt, n_tokens):
    payload = {"n":1,
               "temperature":1,
               "top_k":50,
               "repetition_penalty":1.02,
               "min_new_tokens":16,
               "max_tokens": n_tokens,
               # "max_new_tokens":n_tokens,
               "model":model_name,
               "prompt":prompt,
               "stream":False}
    completion = requests.post("http://localhost:8000/v1/completions/",
                               data=json.dumps(payload))
    return completion.json()["choices"][0]["text"]

async def main():
    parser = ArgumentParser()
    parser.add_argument("dictionary")
    parser.add_argument("--generator", default="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ")
    parser.add_argument("--evaluator", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    generate_fn = partial(generate_from_vllm, args.generator)
        
    synthesis_prompt = """<s> [INST] The following is a list of parts that are meant to uniquely identify a particular word. Give me your top 5 hypothesis for what the word is and why.

{prompt}

Remember that each hypothesis should be a single word, not a phrase.[/INST]1. {response} -"""
    
    # Prefilter dictionary
    # Debian dictionary path
    with open(args.dictionary) as infile:
        part_lists = json.load(infile)
    # Pool so TQDM can give estimated completion time
    word_batches = [i for i in batched(part_lists.keys(), args.batch_size)]
    word_synthesis = {}
    
    for i, word_batch in enumerate(tqdm(word_batches)):
        batch_pairs = [(word, part_lists[word]) for word in word_batch]
        prompts = [synthesis_prompt.format(prompt=pair[1],
                                           response=pair[0].capitalize())
                   for pair in batch_pairs]
        waiting_gens = []
        for prompt in prompts:
            waiting_gens.append(asyncio.create_task(generate_fn(prompt, 256)))
        lists = []
        for waiting_gen in waiting_gens:
            await waiting_gen
            lists.append(waiting_gen.result().strip())
        for j in range(len(word_batch)):
            word_synthesis[batch_pairs[j][0]] = (batch_pairs[j][1], lists[j])
            print(f"{batch_pairs[j][0]}")
            print(lists[j])
        #evaluation_prompt = prepare_rubric("prompts/parts_list_rubric.txt")
        #score_prompt_fn = partial(make_score_prompt_fn, evaluation_prompt)
        # Mixtral Instruct suffix
        #score_prompt_fns = [partial(score_prompt_fn, "[/INST]"),]
        # Change name to avoid overwriting global baseline evaluate_fn partial
        #score_fn = partial(evaluate_fn, score_prompt_fns)
        #scores = score_fn([[word_batch[i], lists[i]] for i in range(args.batch_size)])
        #for i, score in enumerate(scores):
        #    if score.item() > 10:
        #        
        #        print(f"{word_batch[i]} accepted")
        #        print(lists[i])
        if i % 100 == 0:
            with open("word_synthesis.json", "w") as outfile:
                json.dump(word_synthesis, outfile)

    with open("word_synthesis.json", "w") as outfile:
        json.dump(word_synthesis, outfile)

if __name__ == "__main__":
    asyncio.run(main())
