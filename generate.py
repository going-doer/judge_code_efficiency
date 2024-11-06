from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import json
from datasets import load_dataset
from typing import Optional, Dict, Sequence
import copy
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data_path", type=str)
ap.add_argument("--extracted_key", type=str, default="input")
ap.add_argument("--output_path", type=str)
ap.add_argument("--model_path", default="deepseek-ai/deepseek-coder-1.3b-instruct")
ap.add_argument("--max_length", type=int, default=2048)
args = ap.parse_args()


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Update the given code to make it more efficient. 
{}
### Response:
'''.format(instruction.strip()).lstrip()

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

model_name=args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
raw_train_datasets = load_dataset(
    'json',
    data_files=f'{args.data_path}',
    split="train",
)

result_dict = {}

for idx, data in enumerate(tqdm(raw_train_datasets)):
    messages=[
        { 'role': 'user', 'content': f"Update the given code to make it more efficient. {data[args.extracted_key]}"}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", max_length=args.max_length).to(model.device)
    try:
        outputs = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=1.0, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    except Exception as e:
        print(e)
        continue

    decoded_outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)

    result_dict[idx] = {
        'decoded_outputs': decoded_outputs,
    }

with open(f'{args.output_path}', 'w') as f:
    json.dump(result_dict, f)
