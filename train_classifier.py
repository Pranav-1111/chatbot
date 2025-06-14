import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Basic dataset
data = [
    {"query": "How can I reset my password?", "intent": "password_reset"},
    {"query": "I forgot my password. How do I recover it?", "intent": "password_reset"},
    {"query": "What is the process for getting a refund?", "intent": "refund_policy"},
    {"query": "I want to return an item. Can I get my money back?", "intent": "refund_policy"},
    {"query": "My account is locked. How can I unlock it?", "intent": "account_help"},
    {"query": "Where is my order? How do I check?", "intent": "order_tracking"},
    {"query": "Can I cancel my subscription online?", "intent": "subscription_cancel"},
    {"query": "How do I stop my subscription?", "intent": "subscription_cancel"},
]

# Expand dataset with variations
more_queries = [{"query": f"{q['query']} Please help!", "intent": q['intent']} for q in data for _ in range(60)]
data.extend(more_queries)

# Intent to label mapping
intent_to_label = {intent: idx for idx, intent in enumerate(set(q['intent'] for q in data))}
label_to_intent = {v: k for k, v in intent_to_label.items()}

# Add numeric labels
for item in data:
    item["label"] = intent_to_label[item["intent"]]

# Split into train/validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_dataset(dataset):
    queries = [item["query"] for item in dataset]
    labels = [item["label"] for item in dataset]
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
    return inputs, labels

train_inputs, train_labels = tokenize_dataset(train_data)
val_inputs, val_labels = tokenize_dataset(val_data)

# Dataset wrapper
class IntentDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = IntentDataset(train_inputs, train_labels)
val_dataset = IntentDataset(val_inputs, val_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(intent_to_label))

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save model
MODEL_SAVE_PATH = "models/bert_intent_classifier"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"✅ Model and tokenizer saved to {MODEL_SAVE_PATH}")
