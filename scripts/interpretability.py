import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import shap
import lime
from lime.lime_text import LimeTextExplainer
import logging
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


model_path = "path/to/your/saved/model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Label mapping (Adjusted based on your trained model's labels)
label_list = ['B-CONTACT_INFO', 'B-LOC', 'B-PRICE', 'B-Product', 'I-LOC', 'I-PRICE', 'I-Product', 'O']
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)} # Also define label2id if needed

# Update model config with correct label mappings
model.config.id2label = id2label
model.config.label2id = label2id


# Load CoNLL data
def load_conll(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(None, 1) 
                token = parts[0]
                label = parts[1] if len(parts) > 1 else 'O'

                current_sentence.append(token)
                current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
    # Add the last sentence if the file doesn't end with an empty line
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels

# SHAP Explainer
def shap_explain(texts):
    def predict_proba(texts):
        predictions = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # Move inputs to the same device as the model
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            predictions.append(probabilities) 
        return np.vstack(predictions)
    logger.info("SHAP explanation function is a placeholder and needs adaptation for token classification.")
    pass # Placeholder


# LIME Explainer
def lime_explain(texts):
    def predictor(texts_list):
        outputs = []
        for text in texts_list:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
             # Move inputs to the same device as the model
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            # Apply softmax to get probabilities for each token
            probabilities = torch.softmax(logits, dim=-1)
            outputs.append(probabilities.mean(dim=1).cpu().numpy()) 
        return np.vstack(outputs) 
    logger.info("LIME explanation function is a placeholder and needs adaptation for token classification.")
    pass 

logger.info("Code block modified to include correct label mappings and notes on SHAP/LIME adaptation.")
logger.info("Remember to update 'model_path' to your local model directory.")