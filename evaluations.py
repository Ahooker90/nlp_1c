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

evaluator = Evaluator()
bleu_list = []
rouge_list =[]
bert_list = []

top_k = [2, 20, 50, 100]
num_beams = [1, 2, 10, 20]
temps = [0.1, 0.5, 1, 1.5]

results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

bleu_list = []
rouge_list = []
bert_list = []
for k in top_k:
    for beam in num_beams:
        for temp in temps:
            # Reset metrics lists for the current combination
            bleu_list = []
            rouge_list = []
            bert_list = []
            for i in range(len(test_dataset[0])):
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
                    top_k=k,
                    num_beams=beam,
                    temperature=temp
                )
                str_1 = tokenizer.decode(output[0], skip_special_tokens=True)
                str_list = [str_0, str_1]
                evaluator.set_strs_list(str_list)
                bleu, rouge, bert = evaluator.PerformEval(verbose=False)
                bleu_list.append(bleu)
                rouge_list.append(rouge['f'])
                bert_list.append(bert[2])  # F1
                    
                torch.cuda.empty_cache()
                gc.collect()
            
            # Store the results
            results[k][beam][temp] = {'BLEU': Average(bleu_list), 'ROUGE': Average(rouge_list), 'BERT': Average(bert_list)}
            if False:                                              
                print(f"Bleu F1 Score: {Average(bleu_list)}")
                print(f"Rouge F1 Score: {Average(rouge_list)}")
                print(f"Bert Score: {Average(bert_list)}")

save_to_pickle(results, './results.pkl')

for top_k, beams_dict in results.items():
    for num_beams, temps_dict in beams_dict.items():
        for temp, metrics_dict in temps_dict.items():
            print(f"top_k={top_k}, num_beams={num_beams}, temperature={temp}")
            print(f"Metrics: {metrics_dict}")
            print("----")

"""
#Output - mistralai/Mistral-7B-v0.1 (Fine Tuned On Instruction Dataset)
top_k=2, num_beams=1, temperature=0.1
Metrics: {'BLEU': 7.430390470204844e-232, 'ROUGE': 0.2824130307105545, 'BERT': tensor([0.8652])}
----
top_k=2, num_beams=1, temperature=0.5
Metrics: {'BLEU': 7.406893239045227e-232, 'ROUGE': 0.273527709382558, 'BERT': tensor([0.8602])}
----
top_k=2, num_beams=1, temperature=1
Metrics: {'BLEU': 7.401038432613348e-232, 'ROUGE': 0.2736192013549203, 'BERT': tensor([0.8634])}
----
top_k=2, num_beams=1, temperature=1.5
Metrics: {'BLEU': 7.370687295476608e-232, 'ROUGE': 0.28648796744010446, 'BERT': tensor([0.8664])}
----
top_k=2, num_beams=2, temperature=0.1
Metrics: {'BLEU': 7.3770056011491e-232, 'ROUGE': 0.25847579462702386, 'BERT': tensor([0.8629])}
----
top_k=2, num_beams=2, temperature=0.5
Metrics: {'BLEU': 7.341341339793714e-232, 'ROUGE': 0.27256614173654864, 'BERT': tensor([0.8625])}
----
top_k=2, num_beams=2, temperature=1
Metrics: {'BLEU': 7.49839866803196e-232, 'ROUGE': 0.26324153223436375, 'BERT': tensor([0.8557])}
----
top_k=2, num_beams=2, temperature=1.5
Metrics: {'BLEU': 7.661105482841675e-232, 'ROUGE': 0.250951943946024, 'BERT': tensor([0.8641])}
----
top_k=2, num_beams=10, temperature=0.1
Metrics: {'BLEU': 7.431422147075249e-232, 'ROUGE': 0.25538541514951674, 'BERT': tensor([0.8514])}
----
top_k=2, num_beams=10, temperature=0.5
Metrics: {'BLEU': 7.387456480544337e-232, 'ROUGE': 0.2398767015477075, 'BERT': tensor([0.8586])}
----
top_k=2, num_beams=10, temperature=1
Metrics: {'BLEU': 7.466567218562048e-232, 'ROUGE': 0.2474745461072945, 'BERT': tensor([0.8566])}
----
top_k=2, num_beams=10, temperature=1.5
Metrics: {'BLEU': 7.610791425664222e-232, 'ROUGE': 0.23265853224452443, 'BERT': tensor([0.8505])}
----
top_k=2, num_beams=20, temperature=0.1
Metrics: {'BLEU': 7.428198831251975e-232, 'ROUGE': 0.2365964848702605, 'BERT': tensor([0.8631])}
----
top_k=2, num_beams=20, temperature=0.5
Metrics: {'BLEU': 7.389582350967661e-232, 'ROUGE': 0.21312293338247168, 'BERT': tensor([0.8607])}
----
top_k=2, num_beams=20, temperature=1
Metrics: {'BLEU': 7.439345574446186e-232, 'ROUGE': 0.22230603332742413, 'BERT': tensor([0.8553])}
----
top_k=2, num_beams=20, temperature=1.5
Metrics: {'BLEU': 7.419867884535988e-232, 'ROUGE': 0.24578187019857545, 'BERT': tensor([0.8555])}
----
top_k=20, num_beams=1, temperature=0.1
Metrics: {'BLEU': 7.421868128026167e-232, 'ROUGE': 0.2632840843497954, 'BERT': tensor([0.8585])}
----
top_k=20, num_beams=1, temperature=0.5
Metrics: {'BLEU': 7.445496306304056e-232, 'ROUGE': 0.24849851435420453, 'BERT': tensor([0.8621])}
----
top_k=20, num_beams=1, temperature=1
Metrics: {'BLEU': 7.643370345978005e-232, 'ROUGE': 0.2631825990246918, 'BERT': tensor([0.8620])}
----
top_k=20, num_beams=1, temperature=1.5
Metrics: {'BLEU': 7.498258554702837e-232, 'ROUGE': 0.2768094165749612, 'BERT': tensor([0.8643])}
----
top_k=20, num_beams=2, temperature=0.1
Metrics: {'BLEU': 7.525132330912834e-232, 'ROUGE': 0.2720541040468578, 'BERT': tensor([0.8664])}
----
top_k=20, num_beams=2, temperature=0.5
Metrics: {'BLEU': 7.327344185704093e-232, 'ROUGE': 0.27766394873688366, 'BERT': tensor([0.8653])}
----
top_k=20, num_beams=2, temperature=1
Metrics: {'BLEU': 7.443175707864735e-232, 'ROUGE': 0.26054447779782497, 'BERT': tensor([0.8605])}
----
top_k=20, num_beams=2, temperature=1.5
Metrics: {'BLEU': 7.420802816372501e-232, 'ROUGE': 0.27306203593517475, 'BERT': tensor([0.8644])}
----
top_k=20, num_beams=10, temperature=0.1
Metrics: {'BLEU': 7.372594092916584e-232, 'ROUGE': 0.23933130223858323, 'BERT': tensor([0.8592])}
----
top_k=20, num_beams=10, temperature=0.5
Metrics: {'BLEU': 7.414559641612102e-232, 'ROUGE': 0.22938146365229195, 'BERT': tensor([0.8597])}
----
top_k=20, num_beams=10, temperature=1
Metrics: {'BLEU': 7.559506758317845e-232, 'ROUGE': 0.2617705696895407, 'BERT': tensor([0.8561])}
----
top_k=20, num_beams=10, temperature=1.5
Metrics: {'BLEU': 7.35076907552968e-232, 'ROUGE': 0.26516324882283526, 'BERT': tensor([0.8580])}
----
top_k=20, num_beams=20, temperature=0.1
Metrics: {'BLEU': 7.275765332339559e-232, 'ROUGE': 0.24926370439037687, 'BERT': tensor([0.8588])}
----
top_k=20, num_beams=20, temperature=0.5
Metrics: {'BLEU': 7.36455349920401e-232, 'ROUGE': 0.24893119422578114, 'BERT': tensor([0.8611])}
----
top_k=20, num_beams=20, temperature=1
Metrics: {'BLEU': 7.488061197996643e-232, 'ROUGE': 0.24450608137221608, 'BERT': tensor([0.8504])}
----
top_k=20, num_beams=20, temperature=1.5
Metrics: {'BLEU': 7.484395049527382e-232, 'ROUGE': 0.22180590458496724, 'BERT': tensor([0.8509])}
----
top_k=50, num_beams=1, temperature=0.1
Metrics: {'BLEU': 7.394773357608902e-232, 'ROUGE': 0.2843690291550708, 'BERT': tensor([0.8605])}
----
top_k=50, num_beams=1, temperature=0.5
Metrics: {'BLEU': 7.456973148907045e-232, 'ROUGE': 0.24940869914680996, 'BERT': tensor([0.8572])}
----
top_k=50, num_beams=1, temperature=1
Metrics: {'BLEU': 7.397012724939378e-232, 'ROUGE': 0.2598542241007427, 'BERT': tensor([0.8569])}
----
top_k=50, num_beams=1, temperature=1.5
Metrics: {'BLEU': 7.364614612518302e-232, 'ROUGE': 0.275726061826587, 'BERT': tensor([0.8624])}
----
top_k=50, num_beams=2, temperature=0.1
Metrics: {'BLEU': 7.585433824429599e-232, 'ROUGE': 0.24683812226120838, 'BERT': tensor([0.8596])}
----
top_k=50, num_beams=2, temperature=0.5
Metrics: {'BLEU': 7.561695051836655e-232, 'ROUGE': 0.2762320134546163, 'BERT': tensor([0.8660])}
----
top_k=50, num_beams=2, temperature=1
Metrics: {'BLEU': 7.466516994760819e-232, 'ROUGE': 0.2799410713496417, 'BERT': tensor([0.8595])}
----
top_k=50, num_beams=2, temperature=1.5
Metrics: {'BLEU': 7.446810591161484e-232, 'ROUGE': 0.24998802226321898, 'BERT': tensor([0.8521])}
----
top_k=50, num_beams=10, temperature=0.1
Metrics: {'BLEU': 7.447688472241227e-232, 'ROUGE': 0.2248131180355458, 'BERT': tensor([0.8569])}
----
top_k=50, num_beams=10, temperature=0.5
Metrics: {'BLEU': 7.52464935091439e-232, 'ROUGE': 0.2546675554707034, 'BERT': tensor([0.8551])}
----
top_k=50, num_beams=10, temperature=1
Metrics: {'BLEU': 7.427191637000915e-232, 'ROUGE': 0.23557449024088908, 'BERT': tensor([0.8629])}
----
top_k=50, num_beams=10, temperature=1.5
Metrics: {'BLEU': 7.386553571651651e-232, 'ROUGE': 0.2516253041673582, 'BERT': tensor([0.8556])}
----
top_k=50, num_beams=20, temperature=0.1
Metrics: {'BLEU': 7.396898686040508e-232, 'ROUGE': 0.23034984245070905, 'BERT': tensor([0.8501])}
----
top_k=50, num_beams=20, temperature=0.5
Metrics: {'BLEU': 7.342446290162588e-232, 'ROUGE': 0.25384366170924005, 'BERT': tensor([0.8631])}
----
top_k=50, num_beams=20, temperature=1
Metrics: {'BLEU': 7.346448065279043e-232, 'ROUGE': 0.2427790148692538, 'BERT': tensor([0.8546])}
----
top_k=50, num_beams=20, temperature=1.5
Metrics: {'BLEU': 7.307565617197931e-232, 'ROUGE': 0.2445975149434692, 'BERT': tensor([0.8553])}
----
top_k=100, num_beams=1, temperature=0.1
Metrics: {'BLEU': 7.347792472616368e-232, 'ROUGE': 0.28059363648000546, 'BERT': tensor([0.8699])}
----
top_k=100, num_beams=1, temperature=0.5
Metrics: {'BLEU': 7.710338437134558e-232, 'ROUGE': 0.2795924074153488, 'BERT': tensor([0.8632])}
----
top_k=100, num_beams=1, temperature=1
Metrics: {'BLEU': 7.515401433444607e-232, 'ROUGE': 0.27057268770985915, 'BERT': tensor([0.8676])}
----
top_k=100, num_beams=1, temperature=1.5
Metrics: {'BLEU': 7.406518503681425e-232, 'ROUGE': 0.27051009298385864, 'BERT': tensor([0.8588])}
----
top_k=100, num_beams=2, temperature=0.1
Metrics: {'BLEU': 7.476379651360174e-232, 'ROUGE': 0.25250550924356396, 'BERT': tensor([0.8578])}
----
top_k=100, num_beams=2, temperature=0.5
Metrics: {'BLEU': 7.56387745092166e-232, 'ROUGE': 0.2514940314765352, 'BERT': tensor([0.8517])}
----
top_k=100, num_beams=2, temperature=1
Metrics: {'BLEU': 7.496919763424898e-232, 'ROUGE': 0.25392413356322435, 'BERT': tensor([0.8500])}
----
top_k=100, num_beams=2, temperature=1.5
Metrics: {'BLEU': 7.445219565004885e-232, 'ROUGE': 0.2768408776449446, 'BERT': tensor([0.8598])}
----
top_k=100, num_beams=10, temperature=0.1
Metrics: {'BLEU': 7.372521873317573e-232, 'ROUGE': 0.23776504317089575, 'BERT': tensor([0.8582])}
----
top_k=100, num_beams=10, temperature=0.5
Metrics: {'BLEU': 7.422670094365701e-232, 'ROUGE': 0.2418477475965625, 'BERT': tensor([0.8581])}
----
top_k=100, num_beams=10, temperature=1
Metrics: {'BLEU': 7.379302381638433e-232, 'ROUGE': 0.2309438278623575, 'BERT': tensor([0.8716])}
----
top_k=100, num_beams=10, temperature=1.5
Metrics: {'BLEU': 7.441426110510765e-232, 'ROUGE': 0.23616369901691525, 'BERT': tensor([0.8630])}
----
top_k=100, num_beams=20, temperature=0.1
Metrics: {'BLEU': 7.329530990050787e-232, 'ROUGE': 0.2390658885503328, 'BERT': tensor([0.8584])}
----
top_k=100, num_beams=20, temperature=0.5
Metrics: {'BLEU': 7.308548875448358e-232, 'ROUGE': 0.25145846840287334, 'BERT': tensor([0.8567])}
----
top_k=100, num_beams=20, temperature=1
Metrics: {'BLEU': 7.326347881604906e-232, 'ROUGE': 0.23789764861043644, 'BERT': tensor([0.8553])}
----
top_k=100, num_beams=20, temperature=1.5
Metrics: {'BLEU': 7.480655842538781e-232, 'ROUGE': 0.26710069792391167, 'BERT': tensor([0.8524])}
"""

















    