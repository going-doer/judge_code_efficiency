from tqdm import tqdm
import argparse
import re
import transformers
import jsonlines
import json
import random

def del_chinese(readData):
    text = re.sub(r'[\u4e00-\u9fff]+', '', readData)
    return text

def del_japanese(text):
    pattern = r'[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]'
    return re.sub(pattern, '', text)

def generate_finetune_data(
        model_max_length=2048, 
        model_name='deepseek-ai/deepseek-coder-1.3b-instruct', 
        jsonl_path="",output_path=""):

    model_max_length=model_max_length - 30

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    output_lines=[]
    code_datas_jsonl = []
    code_ref_jsonl = []
    with jsonlines.open(jsonl_path) as f:
        idx=-1
        cnt = 0
        for line in f.iter():
            idx += 1
            label = random.randint(0,1)
            label_gt = line['generated_answers_0_is_improve']

            code_v0 = line['input']
            code_v0 = del_chinese(code_v0)
            code_v0 = del_japanese(code_v0)

            code_v1 = line['generated_answers_0'] 
            code_v1 = del_chinese(code_v1)
            code_v1 = del_japanese(code_v1)
            
            improve_diff = line['improve_diff'] 

            if label_gt == False:
                temp = code_v0
                code_v0 = code_v1
                code_v1 = temp

            prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Given a selection of code, determine which one is the most efficient in computing. 
A. {code_v0}
B. {code_v1}
### Response:"""
            
            # truncation for longer input
            encoded = tokenizer(prompt)

            while len(encoded['input_ids'])>model_max_length:
                code_v0 = code_v0[:-10]
                code_v1 = code_v1[:-10]
                prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Given a selection of code, determine which one is the most efficient in computing. 
A. {code_v0}
B. {code_v1}
### Response:"""
                
                encoded = tokenizer(prompt)
            
            if label == 0:
                code_datas_jsonl.append({
                    'instruction': f"Given a selection of code, determine which one is the most efficient in computing. \nA. {code_v1} \nB. {code_v0}",
                    'output': 'A',
                    'improve_diff': improve_diff, 
                    'is_improve': line['generated_answers_0_is_improve']
                })
            else:
                code_datas_jsonl.append({
                    'instruction': f"Given a selection of code, determine which one is the most efficient in computing. \nA. {code_v0} \nB. {code_v1}",
                    'output': 'B',
                    'improve_diff': improve_diff, 
                    'is_improve': line['generated_answers_0_is_improve']
                })
    
    with open(output_path, 'w') as f:
        for code_json in code_datas_jsonl:
            json.dump(code_json, f)
            f.write("\n")

def convert_deepseekcoder_output_to_eval_input(
    lang="python",
    input_path="",
    result_path="", 
    output_path=""): 

    result_self_improve = {}
    idx = 0
    if result_path.endswith('.jsonl'):
        with jsonlines.open(result_path) as f:
            for line in f.iter():
                result_self_improve[f"{idx}"] = {
                    'decoded_outputs': line['decoded_outputs'], 
                    'label':'',
                    'idx': idx,
                }
                idx += 1

    elif result_path.endswith('json'):
            with open(result_path) as f:
                result_self_improve = json.load(f)

    else:
        raise Exception('wrong file extension (json or jsonl)')

    

    original_input_lines = []
    original_input_lines_dict = {}
    with jsonlines.open(input_path) as reff:
        tidx = 0
        for line in reff.iter():
            original_input_lines.append(line)
            original_input_lines_dict[tidx] = line
            tidx += 1


    result_self_improve_codes_lst = []
    empty_cnt = 0

    idx = 0
    for key, item in tqdm(result_self_improve.items()):
        if key == "total": 
            continue
        item['idx'] = idx
        
        result_self_improve_code={
            'generated_answers': [],
            'input': original_input_lines_dict[int(item['idx'])]['input'],
            'answer': original_input_lines_dict[int(item['idx'])]['target'] 
        }
        
        generated_outputs = item['decoded_outputs']
        try:
            start_str = f"```{lang}"
            end_str = "```"

            if lang == "cpp":
                if start_str not in generated_outputs:
                    start_index = generated_outputs.index(f"```c")
                else:
                    start_index = generated_outputs.index(start_str)
            else: # python
                start_index = generated_outputs.index(start_str)

            end_index = generated_outputs.rindex(end_str)
            generated_outputs_code = generated_outputs[start_index+len(start_str):end_index]

            result_self_improve_code['generated_answers'].append(generated_outputs_code)
        except Exception as e:
            print(e)

            result_self_improve_code['generated_answers'].append(generated_outputs)
        
        if result_self_improve_code['generated_answers'][-1] == "":
            empty_cnt += 1

        
        result_self_improve_codes_lst.append(result_self_improve_code)

        idx += 1

    assert len(result_self_improve_codes_lst) == len(original_input_lines), "length is not equal"

    with open(output_path, 'w') as f:
        for my_data in result_self_improve_codes_lst:
            json.dump(my_data, f)
            f.write("\n") 


ap = argparse.ArgumentParser()
ap.add_argument("--jsonl_path", type=str, default="")
ap.add_argument("--model_path", default="deepseek-ai/deepseek-coder-1.3b-instruct")
ap.add_argument("--max_length", type=int, default=2048)
ap.add_argument("--result_path", type=str)
ap.add_argument("--output_path", type=str)
ap.add_argument("--input_path", type=str, default="")
ap.add_argument("--lang", type=str, default="python")

args = ap.parse_args()

# generate_finetune_data(
#         model_max_length=args.max_length, 
#         model_name=args.model_path, 
#         jsonl_path=args.jsonl_path,
#         output_path=args.output_path)

convert_deepseekcoder_output_to_eval_input(
    lang=args.lang,
    input_path=args.input_path,
    result_path=args.result_path,
    output_path=args.output_path,
)