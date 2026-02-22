
import os
import pandas as pd
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = "google/pegasus-cnn_dailymail" # Using a popular pre-trained base
# SAMSUM usually works better with pegasus-samsum, but the prompt says "Fine-tune on SAMSUM train set" 
# implying we start from a base model or cnn_dailymail. 
# The workflow mentions 'google/pegasus-cnn_dailymail'.

def prepare_data():
    data_path = os.path.join("artifacts", "data")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    
    # Fill NAs
    train_df = train_df.fillna("")
    val_df = val_df.fillna("")
    test_df = test_df.fillna("")
    
    # Create Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

def train_pegasus():
    logging.info("Starting Pegasus Training...")
    
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    
    dataset = prepare_data()
    
    def preprocess_function(examples):
        inputs = examples["dialogue"]
        targets = examples["summary"]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    output_dir = os.path.join("artifacts", "models", "pegasus")
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2, # Small batch size for standard GPU/CPU
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logging.info("Training...")
    trainer.train()
    
    logging.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_pegasus()
