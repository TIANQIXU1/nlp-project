# nlp-project
Code and data for the paper:
To improve the translation quality of medical terminologies from English to Chinese, we compare the fine tuning result of a Neural Machine
Translation (NMT) model using the ECCParaCorp data, with a large language model(Qwen2.5) using one-shot translation prompts within the same ECCParaCorp data.

## Data (training and test)
### Training data
In the training of the NMT model, we utilized the dataset: [EN-to-ZH ECCParaCorp](https://github.com/TIANQIXU1/nlp-project/blob/main/data/Ecc%20train%20phrases.csv).

In the training of LLM, we utilized the same dataset to generate prompts.
### Test data
In the test, we used our manually annotated EN cancer terminology (193 words and 109phrases) with the ZH reference: [test en_zh.json](https://github.com/TIANQIXU1/nlp-project/blob/main/data/test%20en_zh.json).

## Fine-tuning opus-mt-en-zh
In the first phase, in-domain data is used to fine-tune the MT model. You can see the script here:[opus-fine_tuning.py](https://github.com/TIANQIXU1/nlp-project/blob/main/opus-fine_tuning.py).
## Fine-tuning Qwen2.5 1.5B
Prompts are created using the 'create_prompt'function
## Inference

### Tokenizers
* **1.5B Qwen2.5 model**

### Translation

## Evaluation
Evaluation was done based on BLEU and human evaluation.
