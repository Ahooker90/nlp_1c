import os
import io
import sys
import time
import json
import pandas as pd
import datasets
from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig,  TaskType
from peft import replace_lora_weights_loftq
from peft import get_peft_model
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from trl import SFTTrainer

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

#base_model = "mistralai/Mistral-7B-v0.1"
#base_model = "meta-llama/Llama-2-7b-hf"
base_model = "microsoft/phi-2"

lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         llm_int8_threshold=200.0)

tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
)

peft_model = get_peft_model(base_model, lora_config)
replace_lora_weights_loftq(peft_model)


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

trainer = SFTTrainer(
    model = peft_model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs= 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "phi_2_outputs",
    ),
)

trainer.train()
peft_model.save_pretrained("./phi_2_output_dir")

text = alpaca_prompt.format(test_dataset[0]['instruction'], test_dataset[0]['input'], '') + EOS_TOKEN

# Tokenize the input text
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate output text using the model
output = peft_model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)


print("Stop!")





















