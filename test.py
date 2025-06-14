import torch
import tokenizer
import model

query = "I need help with my password"
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
predicted_label = torch.argmax(logits, dim=1).item()
print(f"Predicted Label: {predicted_label}")
