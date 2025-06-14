import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Label to intent mapping (must match training)
label_to_intent = {
    0: "password_reset",
    1: "refund_policy",
    2: "account_help",
    3: "order_tracking",
    4: "subscription_cancel",
}

MODEL_SAVE_PATH = "models/bert_intent_classifier"

if not os.path.exists(MODEL_SAVE_PATH) or not os.listdir(MODEL_SAVE_PATH):
    raise FileNotFoundError(f"Model directory '{MODEL_SAVE_PATH}' is empty or missing. Retrain the model.")

tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.eval()

def predict_intent(query, debug=False):
    if not query.strip():
        return "❌ Invalid input. Please enter a valid query."

    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    if debug:
        print(f"Logits: {logits}")
        print(f"Predicted Label Index: {predicted_label}")

    return label_to_intent.get(predicted_label, "❓ Unknown Intent")

if __name__ == "__main__":
    test_queries = [
        "How do I reset my password?",
        "Can I cancel my subscription?",
        "I need help with my account.",
        "What is your refund policy?",
        "Where can I update my account details?",
        "How long does shipping take?"
    ]

    for query in test_queries:
        intent = predict_intent(query, debug=True)
        print(f"🟢 Query: {query}\n🔹 Predicted Intent: {intent}\n")
