import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from index import preprocess_text, embedder, get_intent_emb, SIM_THRESHOLD, FALLBACK_TAG

# --- Load intents ---
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# --- Build mapping of expected sentences to intents ---
# Add your own sentences to test more thoroughly
test_samples = [
    ("Hi there!", "greeting"),
    ("I feel anxious lately", "anxiety"),
    ("I am very depressed today", "depression"),
    ("I'm stressed about work", "stress"),
    ("I feel lonely and sad", "loneliness"),
    ("I feel worthless", "self_worth"),
    ("I cannot sleep properly", "sleep"),
    ("I want to be productive", "motivation"),
    ("I feel angry", "anger"),
    ("I feel thankful", "gratitude"),
    ("Bye", "goodbye"),
    ("Random gibberish text", "fallback")
]

# --- Run test ---
print("Running chatbot intent tests...\n")

for sentence, expected_intent in test_samples:
    processed = preprocess_text(sentence)
    emb = embedder.encode([processed], convert_to_tensor=True)
    pred_intent, conf = get_intent_emb(emb[0])
    
    # Apply threshold
    if conf < SIM_THRESHOLD:
        pred_intent = FALLBACK_TAG

    correct = "✅" if pred_intent == expected_intent else "❌"
    
    print(f"Sentence: {sentence}")
    print(f"Expected Intent: {expected_intent}")
    print(f"Predicted Intent: {pred_intent} (confidence={conf:.2f}) {correct}\n")
