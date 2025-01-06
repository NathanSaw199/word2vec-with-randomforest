from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer
from transformers import TrainingArguments
dataset = load_dataset('imdb')

# print(dataset)
print(dataset["train"][0])
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# print(tokenized_datasets)
# print(tokenized_datasets["train"][0])

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    eval_strategy ="epoch",     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=1,              # Number of training epochs
    weight_decay=0.01,               # Strength of weight decay
)

print(training_args)

#initalize the model 
#load pretrained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


#initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)


#train the model

print(trainer.train())

#evaluate the model

results = trainer.evaluate()
print(results)



# Save the model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-tokenizer')
