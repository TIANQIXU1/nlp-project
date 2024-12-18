
## Fine-tuned model BLEU score 44.07


import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb logging
import pandas as pd
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import DataCollatorForSeq2Seq
from sacrebleu import corpus_bleu

# Load English-Chinese dataset
data_path = "/content/drive/MyDrive/machine_translation_en_ch/cancer_data/en_zh_dictionary.csv"
df = pd.read_csv(data_path)
df = df.rename(columns={"en_description": "source", "zh_description": "target"})

# Load tokenizer and model
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["source"], text_target=examples["target"], truncation=True, padding=True, max_length=128)

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into training, validation, and testing
train_test_split = tokenized_dataset.train_test_split(test_size=0.4, seed=42)  # 60% training, 40% for test + dev
test_dev_split = train_test_split["test"].train_test_split(test_size=0.25, seed=42)  # Split 40% into 30% test, 10% dev
train_dataset = train_test_split["train"]
test_dataset = test_dev_split["test"]
dev_dataset = test_dev_split["train"]

# Data collator for dynamic batching by sequence length
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

def lr_scheduler(optimizer, num_warmup_steps=4000, d_model=512):
    def lr_lambda(step):
        if step == 0:
            return 0  # Handle step 0 to avoid division by zero
        arg1 = step ** -0.5
        arg2 = step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)
    return LambdaLR(optimizer, lr_lambda)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/translation_model_output_en_zh_dict",
    eval_strategy="epoch",  # Updated from evaluation_strategy
    save_strategy="epoch",
    learning_rate=1e-4,  # Placeholder, overridden by scheduler
    num_train_epochs=420,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=4000,
    weight_decay=0.01,
    logging_dir="/content/drive/MyDrive/logging",
    save_total_limit=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision if on GPU
    gradient_accumulation_steps=2,  # Simulates a batch size of 32
)

# Custom trainer with label smoothing for regularization
class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,  # Use dev dataset for evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler(optimizer)),  # Optimizer with custom scheduler
)

# Train the model with automatic checkpoint resumption
trainer.train(resume_from_checkpoint=True)

# Evaluate BLEU score
def evaluate_bleu(trainer, eval_dataset):
    predictions = trainer.predict(eval_dataset)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    bleu_score = corpus_bleu(decoded_preds, [decoded_labels]).score
    print(f"BLEU Score: {bleu_score:.2f}")
    return bleu_score

bleu_score = evaluate_bleu(trainer, test_dataset)  # Use test dataset for final evaluation

# Save model and tokenizer
model.save_pretrained("/content/drive/MyDrive/translation_model_output_system1_en_zh_dict")
tokenizer.save_pretrained("/content/drive/MyDrive/translation_model_output_system1_tokenizer_en_zh_dict")
