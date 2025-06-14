from intent_classifier import classify_user_query

if __name__ == "__main__":
    while True:
        user_input = input("🗣️ Your query (type 'exit' to stop): ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        intent = classify_user_query(user_input)
        print(f"🎯 Predicted Intent: {intent}\n")
