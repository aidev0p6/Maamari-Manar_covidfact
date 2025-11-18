import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
# from transformers.trainer import Trainer
# from transformers.training_args import TrainingArguments 
from datasets import Dataset
import evaluate
import numpy as np

# 1. Load data from the TSV files
def load_tsv_data(split='train'):
    """
    Load data from the RTE-covidfact TSV files.
    Handle different column names across files.
    """
    file_path = f"RTE-covidfact/{split}.tsv"
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Loaded {split} set from {file_path} with {len(df)} examples.")
        print(f"Columns found: {list(df.columns)}")
        
        # Handle different column names across files
        # For 'entailment' column (dev.tsv uses 'entailment', train.tsv uses 'label')
        if 'entailment' in df.columns:
            df = df.rename(columns={'entailment': 'label'})
            print("Renamed 'entailment' column to 'label'")
        elif 'label' not in df.columns:
            print(f"Error: Could not find label/entailment column in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            exit(1)
        
        # For evidence/claim columns (dev.tsv has 'Sentence2' with capital S)
        if 'sentence1' in df.columns and 'Sentence2' in df.columns:
            df = df.rename(columns={'sentence1': 'evidence', 'Sentence2': 'claim'})
            print("Renamed 'sentence2' to 'claim'")
        elif 'sentence1' in df.columns and 'sentence2' in df.columns:
            df = df.rename(columns={'sentence1': 'evidence', 'sentence2': 'claim'})
        
        # Map 'entailment' to 1, 'not_entailment' to 0
        df['label'] = df['label'].map({'entailment': 1, 'not_entailment': 0})
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
        exit(1)

print("Loading datasets...")
train_df = load_tsv_data('train')
eval_df = load_tsv_data('dev')  # Using 'dev' as the validation set

# 2. Format the input text for RoBERTa
# The standard format for entailment is: "<s> CLAIM </s></s> EVIDENCE </s>"
print("Formatting text...")
# We use 'claim' (sentence2) first, then 'evidence' (sentence1), as is standard for hypothesis/premise.
train_df['text'] = train_df['claim'] + " </s></s> " + train_df['evidence']
eval_df['text'] = eval_df['claim'] + " </s></s> " + eval_df['evidence']

# 3. Initialize Model and Tokenizer
model_name = "roberta-base"
print(f"Loading model and tokenizer: {model_name}...")
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing data...")
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 5. Set up metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": acc}

# 6. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./final_results",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.1,
    learning_rate=1,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",          # This should work in 4.57.1
    save_strategy="epoch",                # This should work in 4.57.1
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=True,         
)

# 7. Create and run Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# 8. Final evaluation on the VALIDATION set (since test has no labels)
print("*** Evaluating on the VALIDATION set ***")
eval_results = trainer.evaluate(tokenized_eval)  # Changed from tokenized_test to tokenized_eval
print(f"\nFinal Validation Results: {eval_results}")

# Save everything
trainer.save_model("./final_covidfact_model")
with open("./final_validation_results.txt", "w") as f:  # Changed filename
    f.write(f"Hugging Face Reproduction - Oracle Setting\n")
    f.write(f"Validation Accuracy: {eval_results.get('eval_accuracy')}\n")  # Changed metric name
    f.write(f"Validation Loss: {eval_results.get('eval_loss')}\n")  # Changed metric name
print("Model and results saved.")

