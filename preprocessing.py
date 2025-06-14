import spacy

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Preprocess the input text by tokenizing, lemmatizing, and removing stop words and punctuations.
    """
    doc = nlp(text)  # Process the text using spaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Test the function
if __name__ == "__main__":
    sample_text = "How can I reset my password quickly?"
    print("Original Text:", sample_text)
    print("Preprocessed Text:", preprocess_text(sample_text))
