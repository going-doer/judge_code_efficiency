[EMNLP Findings 2024] Rethinking Code Refinement: Learning to Judge Code Efficiency

# Rethinking Code Refinement: Learning to Judge Code Efficiency

Official Code Repository for the paper - Rethinking Code Refinement: Learning to Judge Code Efficiency (EMNLP Findings 2024): https://arxiv.org/abs/2410.22375.

## Requirements

* Python 3.11.8
* PyTorch 2.2.2
* transformers 4.39.3

## Datasets
Training and validation data are in the [data](https://github.com/going-doer/judge_code_efficiency/data) folder. 
Run the command below to unzip the datasets.

Our project is based on IBM CodeNet, and we utilized the preprocessed data format provided by the [PIE project](https://github.com/LearningOpt/pie). 
To generate refined code using LLMs, please refer to [Self Refined Code by LLMs](#self-refined-code-by-llms) below.

```sh
$ python ./scripts/preprocess_data.sh
```

## Train
Run the command below, in order to train the classifier to judge code efficiency in Python and C++ code.

```sh
$ python ./scripts/train.sh
```

## Run and Evaluation
The following command runs experiments for the trained classifier on Python datasets.
```sh
$ sh ./scripts/run.sh
```


## Self Refined Code by LLMs
Run the command below to generate refined code using an LLM. This script utilizes DeepSeekCode-1.3b as the language model.

```sh
$ python ./scripts/generate.sh
```
