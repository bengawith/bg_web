import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # Replace with your preferred pre-trained conversational model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Add a distinct [PAD] token and resize the model's embeddings
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load your training dataset
data_path = "training_data.json"  # Path to your JSON training dataset
dataset = load_dataset("json", data_files=data_path)

# Preprocess the dataset
def preprocess_data(examples):
    inputs = examples["prompt"]
    targets = examples["completion"]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["prompt", "completion"])

# Split dataset into training and validation sets
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
val_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,  # Lower learning rate for better convergence
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,  # Train for more epochs
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model training complete and saved to './fine_tuned_model'.")

def generate_response(user_message):
    inputs = tokenizer(
        user_message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    reply_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Test the model
print(generate_response("Who is Benjamin Gawith?"))
