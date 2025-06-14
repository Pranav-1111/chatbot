from inference import predict_intent

test_queries = [
    "How do I reset my password?",
    "Can I cancel my subscription?",
    "I need help with my account.",
    "What is your refund policy?",
    "Where is my order?",
]

for query in test_queries:
    print(f"Query: {query} \nPredicted Intent: {predict_intent(query)}\n")
