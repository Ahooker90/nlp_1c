import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import io
import sys
import time
import json
import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from collections import defaultdict
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from Evaluator import Evaluator
import cloudpickle as pickle

def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def Average(lst): 
    return sum(lst) / len(lst)

token_model_dir = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(token_model_dir,padding_side='left')
model_dir = "/work/ree398/LLM-Workshop/mistral_7b_output_dir"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

generator = TextGenerationPipeline(
    model=model, tokenizer=tokenizer)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


dataset_path = "/work/ree398/LLM-Workshop/alpaca_data.json"
data = jload(dataset_path)
dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset.map(formatting_prompts_func, batched=True)

train_dataset, test_dataset = train_test_split(dataset["train"], test_size=0.2, random_state=42)
train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=train_dataset))
test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=test_dataset))

ev = Evaluator()
for i in range(20):#Test 20 samples for human evaluation
    str_0 = test_dataset[i]['output']
    text = alpaca_prompt.format(test_dataset[i]['instruction'], test_dataset[i]['input'], '') + EOS_TOKEN
    tokens = tokenizer(text, return_tensors="pt", padding="longest", pad_to_multiple_of=8)
    attention_mask = tokens['attention_mask'].to('cuda')
    input_ids = tokens['input_ids'].to('cuda')
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
    )
    str_1 = tokenizer.decode(output[0], skip_special_tokens=True)
    #print(f"Text: {text}")
    #print(f"Ground Truth: {str_0}")
    print(f"~~~~~~~~~~~~~~~~~~~~~{i}~~~~~~~~~~~~~~~~~")
    print(f"Text: {text}")
    print(f"LLM Output: {str_1}")
    str_list = [str_0, str_1]

























