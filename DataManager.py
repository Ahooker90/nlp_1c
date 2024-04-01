import os
import io
import sys
import time
import json
import pandas as pd
import datasets
from transformers import AutoTokenizer
import torch
import logging
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
class DataManager:
    
    def __init__(self,model, data_path, load_data, load_data_path,load_jsonl):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if load_data:
            if load_jsonl:
                self.loaded_data = self.load_jsonl(load_data_path)
            else:
                #self.loaded_data = self.jload(load_data_path)
                self.loaded_data = self.load_dataset_from_pickle(load_data_path)
        else:
            self.save_bool = True
            self.save_struct_data = False #not working
            self.device = self._allocate_device()
            self.raw_data = self.jload(data_path)
            self.test_data, self.train_data, self.train_struct, self.test_struct = self._split_to_train_test()
            if self.save_bool == True:
                self.save_test_train(self.test_data, self.train_data,self.test_struct, self.train_struct)
            self.struct_dataset
            
    def save_dataset_with_pickle(self,dataset, file_path):
        """
        Saves a dataset to a file using pickle.
    
        Parameters:
        - dataset (datasets.Dataset): The dataset to be saved.
        - file_path (str): The file path where the dataset should be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)
    
        print(f"Dataset saved to: {file_path}")
        
    def load_dataset_from_pickle(self,file_path):
        """
        Loads a dataset from a pickle file.
    
        Parameters:
        - file_path (str): The file path from which to load the dataset.
        
        Returns:
        - The loaded dataset.
        """
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        
        return dataset

            
    
    def get_loaded_data(self):
        return self.loaded_data
    
    def get_train_path(self):
        script_dir = os.path.dirname(__file__)
        return os.path.join(script_dir, 'train_data.json')
    
    def get_test_path(self):
        script_dir = os.path.dirname(__file__)
        return os.path.join(script_dir, 'test_data.json')
    
    def save_test_train(self,test_data, train_data,test_struct,train_struct):
        if self.save_bool:#TODO update to include jsonl and json saving
            script_dir = os.path.dirname(__file__)
            train_path = os.path.join(script_dir, 'train_data.pkl')
            test_path = os.path.join(script_dir, 'test_data.pkl')
            test_struct_path = os.path.join(script_dir, 'test_struct.pkl')
            train_struct_path = os.path.join(script_dir, 'train_struct.pkl')
            self.save_dataset_with_pickle(test_data, test_path)
            self.save_dataset_with_pickle(train_data, train_path)
            self.save_dataset_with_pickle(test_struct, test_struct_path)
            self.save_dataset_with_pickle(train_struct,train_struct_path)
    
    def load_test_train(self,test_path, train_path):
        test_data = self.load_dataset_from_pickle(test_path)
        train_data = self.load_dataset_from_pickle(train_path)
        return test_data, train_data
    
    
    def get_test_train_data(self):
        return self.test_data, self.train_data
    
    def _get_prompt_with_input(self):
        prompt_temp_w_input = """
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input}
            
            ### Response:
            """
        return prompt_temp_w_input
    
    def _get_prompt_without_input(self):
        prompt_temp_w_out_input = """
            Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            """
        return prompt_temp_w_out_input
    
    def _structure_data_with_prompts(self):
        data = self.raw_data
        num_examples = len(data)
        struct_dataset = []
        for i in range(num_examples):
          instruction = data[i]['instruction']
          output = data[i]["output"]
          if data[i]['input'] != '':
            input = data[i]['input']
            text_w_prompt_temp = self._get_prompt_with_input().format(instruction=instruction, input=input)
          else:
            text_w_prompt_temp = self._get_prompt_without_input().format(instruction=instruction)
          
          struct_dataset.append({"instruction": text_w_prompt_temp, "output": output})
          if self.save_struct_data:
              script_dir = os.path.dirname(__file__)
              file_path = os.path.join(script_dir, 'struct_data.json')
              # Open a file to write
              with open(file_path, 'w') as file:
                  # Use json.dump() to write the list to the file
                  json.dump(struct_dataset, file)
          self.struct_dataset = struct_dataset
        return struct_dataset
     
    def get_struct_data_with_prompts(self, load):
        if load:#this isnt right?
            self.jload('/work/ree398/nlp/LLM-Workshop/src/llm-workshop/struct_data.json')
        return self.struct_dataset
        
    def _tokenize_function(self,examples):
        # Concatenate 'instruction' and 'output' fields
        texts = [instr + out for instr, out in zip(examples['instruction'], examples['output'])]
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",  
            truncation=True,       
            max_length=64,         
            return_tensors="np"    
        )
        
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs
    
    def _tokenize_data(self):
        data=self._structure_data_with_prompts()
        train_dataset, test_dataset = train_test_split(data, test_size=0.1, random_state=42)
        train_d = datasets.Dataset.from_pandas(pd.DataFrame(train_dataset))
        test_d = datasets.Dataset.from_pandas(pd.DataFrame(test_dataset))
        if False:
            script_dir = os.path.dirname(__file__)
            train_path = os.path.join(script_dir, 'train_raw.json')
            test_path = os.path.join(script_dir, 'test_raw.json')
            train_d.to_json(train_path, orient='records', lines=False)
            test_d.to_json(test_path , orient='records', lines=False)
            print("Save 1")

        columns_to_remove = ['instruction', 'output']
        
        tokenized_train = train_d.map(
            self._tokenize_function,
            batched=True,
            batch_size=1,
            drop_last_batch=True,
            remove_columns=columns_to_remove
        )
        tokenized_test = test_d.map(
            self._tokenize_function,
            batched=True,
            batch_size=1,
            drop_last_batch=True,
            remove_columns=columns_to_remove
        )

        if False:
            script_dir = os.path.dirname(__file__)
            train_path = os.path.join(script_dir, 'train_data.json')
            test_path = os.path.join(script_dir, 'test_data.json')
            tokenized_train.to_json(train_path, orient='records', lines=False)
            tokenized_train.to_json(test_path , orient='records', lines=False)
            print("Save 2")
        return tokenized_train, tokenized_test, train_d, test_d
    
    def _split_to_train_test(self):
        tokenized_train, tokenized_test, train_d, test_d = self._tokenize_data()

        #if self.save_bool:#TODO update to include jsonl and json saving
            #script_dir = os.path.dirname(__file__)
            #train_path = os.path.join(script_dir, 'train_data.json')
            #test_path = os.path.join(script_dir, 'test_data.json')
            #train_dataset.to_json(train_path, orient='records', lines=False)
            #test_dataset.to_json(test_path , orient='records', lines=False)
            #train_dataset = pd.read_json(train_path, orient='records', lines=True)
            #test_dataset = pd.read_json(test_path, orient='records', lines=True)
            #self.jdump(train_dataset.to_dict(),train_path)
            #self.jdump(test_dataset.to_dict(), test_path)
        return tokenized_test, tokenized_train,train_d, test_d
    
    def _allocate_device(self):
        logger = logging.getLogger(__name__)
        device_count = torch.cuda.device_count()
        if device_count > 0:
            logger.debug("Select GPU device")
            device = torch.device("cuda")
        else:
            logger.debug("Select CPU device")
            device = torch.device("cpu")
        return device
    
    def _make_w_io_base(self,f, mode: str):
        if not isinstance(f, io.IOBase):
            f_dirname = os.path.dirname(f)
            if f_dirname != "":
                os.makedirs(f_dirname, exist_ok=True)
            f = open(f, mode=mode)
        return f


    def _make_r_io_base(self,f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode)
        return f

    def jdump(self,obj, f, mode="w", indent=4, default=str):
        """Dump a str or dictionary to a file in json format.

        Args:
            obj: An object to be written.
            f: A string path to the location on disk.
            mode: Mode for opening the file.
            indent: Indent for storing json dictionaries.
            default: A function to handle non-serializable entries; defaults to `str`.
        """
        f = self._make_w_io_base(f, mode)
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=indent, default=default)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")
        f.close()

    def load_jsonl(self,file_path, mode='r'):
        """Load a .jsonl file into a list of dictionaries."""
        data = []
        with open(file_path, mode) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def jload(self,f, mode="r"):
        """Load a .json file into a dictionary."""
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict
