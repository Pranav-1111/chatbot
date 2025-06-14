from inference import predict_intent

def classify_user_query(user_input):
    return predict_intent(user_input)
