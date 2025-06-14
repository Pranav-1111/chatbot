# 🤖 AI Chatbot

This project solves the problem of handling repetitive support queries using NLP. It classifies user intent from chat using BERT and provides relevant responses via a Flask API. Useful for support automation in e-commerce and SaaS products.


# 🤖 AI Chatbot using BERT for Intent Classification

This project is an AI-powered chatbot built using **BERT (Bidirectional Encoder Representations from Transformers)** to classify user queries into predefined intents and respond accordingly. It uses a **Flask backend**, a simple **HTML+JS frontend**, and is trained with HuggingFace's `transformers` library.

---

## 🚀 Features

- BERT-based intent classification
- Trained on 500+ real-world queries
- Handles multiple intents like password reset, refunds, account help, order tracking, and subscription cancellation
- Flask API backend
- Lightweight frontend (HTML, CSS, JS)
- Easily extendable with more intents

---

## 📁 Project Structure

├── app.py # Flask backend
├── train_classifier.py # Training script
├── inference.py # Manual prediction testing
├── templates/
│ └── index.html # Chatbot UI
├── static/
│ └── style.css # Styling for chatbot
├── models/
│ └── bert_intent_classifier/ # Saved model and tokenizer
├── chatbot_env/ # (Optional) Virtual environment
├── requirements.txt # Python dependencies
└── README.md # You're reading it

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/bert-chatbot.git
cd bert-chatbot
2. Create Virtual Environment
python -m venv chatbot_env
source chatbot_env/bin/activate   # On Windows: chatbot_env\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model (Optional if already trained)
python train_classifier.py
5. Run the Flask App
python app.py
Open your browser and visit:
http://127.0.0.1:5000

💡 Sample Intents
Intent	Sample Queries
password_reset	"How do I reset my password?"
refund_policy	"Can I get a refund?"
account_help	"My account is locked"
order_tracking	"Where is my order?"
subscription_cancel	"I want to cancel my subscription"

🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-lightgrey?logo=flask)
![HuggingFace](https://img.shields.io/badge/Transformers-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)
HTML, CSS, JavaScript

🧠 Future Improvements
Add support for LLM-based answers (like GPT)

Use SQLite or MongoDB for chat history

Improve UI using React or Streamlit

Deploy on Render/Heroku or Dockerize

# 🤖 Chabot Demo
![image](https://github.com/user-attachments/assets/b5567b96-d990-4bbd-92e4-937ff9f23807)


🤝 Contributing
Pull requests are welcome! If you find a bug or want to improve something, open an issue first.

📜 License
MIT License © 2025 Pranav Bhatt
---

Would you like me to also generate a `requirements.txt` or `.gitignore`?
