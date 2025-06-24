# just a simple white space tokenizer for Amharic text
def tokenize_amharic_text(text):
    if not text:
        return []
    return text.split()
