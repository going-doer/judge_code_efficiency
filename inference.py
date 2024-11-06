from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch
import json
from datasets import load_dataset
from typing import Optional, Dict, Sequence
import copy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="./data")
ap.add_argument("--output_dir", type=str, default="./output")
ap.add_argument("--save_postfix", type=str, default="")
ap.add_argument("--data_split", required=True, type=str, help="data split")
ap.add_argument("--ckpt_dir", required=True) # "ckpts/~"
ap.add_argument("--prediction_type", default="classification")
ap.add_argument("--model_path", default="deepseek-ai/deepseek-coder-1.3b-instruct")
args = ap.parse_args()


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
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


tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(f"{args.ckpt_dir}",
                                             trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

raw_train_datasets = load_dataset(
    'json',
    data_files=f'{args.data_dir}/{args.data_split}.jsonl',
    split="train",
)

cnt =0
prob_cnt = 0
regress_cnt = 0
result_dict = {}

y_test = []
pred = []
pred_float = []

for idx, data in enumerate(raw_train_datasets):    
    messages=[
        { 'role': 'user', 'content': data['instruction']}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", max_length=2048).to(model.device)
    try:
        outputs_dict = model.generate(inputs, max_new_tokens=10, do_sample=False, top_k=50, top_p=1.0, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, output_scores=True, output_logits=True, return_dict_in_generate=True)
        outputs = outputs_dict.sequences
    except Exception as e:
        print(e)
        continue

    # Calculate probabilities for each token
    token_probs = torch.nn.functional.softmax(outputs_dict.logits[0], dim=-1).squeeze(0)
    A_prob = token_probs[tokenizer("A").input_ids[-1]]
    B_prob = token_probs[tokenizer("B").input_ids[-1]]
    
    prob_outputs = "B"
    if A_prob > B_prob:
        prob_outputs = "A"
    
    decoded_outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)
    decoded_outputs = decoded_outputs.split('\n')[0]
    label = data['output']

    result_dict[idx] = {
        'decoded_outputs': decoded_outputs,
        'prob_outputs': prob_outputs,
        'label': label
    }
    if 'improve_diff' in data:
        result_dict[idx]['improve_diff'] = data['improve_diff']

    if 'is_improve' in data:
        result_dict[idx]['is_improve'] = data['is_improve']

    pred.append(decoded_outputs)

    if decoded_outputs==label:
        cnt += 1
    else:
        print(f"{idx}\t{decoded_outputs}\t{label}")

    if prob_outputs == label:
        prob_cnt += 1

    if args.prediction_type == "regression":
        label = float(label)
        try:
            pred_value = float(decoded_outputs)
        except:
            pred_value = 1.0
            print(f"{decoded_outputs}\t{pred_value}")

        pred_float.append(pred_value)
        

        if label >=1.0 and pred_value >= 1.0:
            regress_cnt += 1
        elif label < 1.0 and pred_value < 1.0:
            regress_cnt += 1
    y_test.append(label)

result_dict['total'] = {
    'total': f"{cnt}/{len(raw_train_datasets)}",
    'prob_total': f"{prob_cnt}/{len(raw_train_datasets)}",
}

if args.prediction_type == "regression":
    result_dict['total']['MAE'] = mean_absolute_error(y_test, pred_float)
    result_dict['total']['MSE'] = mean_squared_error(y_test, pred_float)
    result_dict['total']['RMSE']  = np.sqrt(result_dict['total']['MSE'])
    result_dict['total']['MSLE']  = mean_squared_log_error(y_test, pred_float)

    result_dict['total']['regression_classify_cnt'] = regress_cnt
    result_dict['total']['regression_classify_acc'] = regress_cnt/len(result_dict)


with open(f'{args.output_dir}/result_{args.save_postfix}.json', 'w') as f:
    json.dump(result_dict, f)

# print(f"{args.data_split}: {cnt}/{len(raw_train_datasets)}")