from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import numpy as np

# Load model and tokenizer
model_path = "models/bert_intent_classifier"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Predict intent from text
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = int(torch.argmax(logits, dim=1))
    return predicted_class_id

# Get intent tag from label
def get_tag_from_label(label_id):
    if "labels" in intents:
        return intents["labels"][label_id]
    else:
        tags = [intent["tag"] for intent in intents["intents"]]
        return tags[label_id]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received JSON:", data)
    user_input = data.get("query", "")  # Fixed here

    if not user_input.strip():
        return jsonify({"response": "Please enter a valid message."})

    try:
        label_id = predict_intent(user_input)
        predicted_tag = get_tag_from_label(label_id)

        matched_intent = next(
            (intent for intent in intents["intents"] if intent["tag"] == predicted_tag),
            None
        )

        if matched_intent and "responses" in matched_intent:
            response = np.random.choice(matched_intent["responses"])
        else:
            response = "Sorry, I don't have a response for that."

        return jsonify({"response": response})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "Oops! Something went wrong."})

if __name__ == "__main__":
    app.run(debug=True)
