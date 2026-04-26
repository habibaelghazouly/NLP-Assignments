import re

def normalize_text(text):
    """
    Lowercase and remove punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text):
    """
    Tokenize text into words.
    """
    return text.split()


def preprocess(text):
    """
    Full preprocessing pipeline.
    """
    text = normalize_text(text)
    tokens = tokenize(text)
    return tokens