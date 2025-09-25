import os
import sys
import json
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sentence_transformers import SentenceTransformer, util
import torch
import random
import datetime
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- NLTK setup ---
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Text cleaning ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", "-")
    return text.strip()

# --- Preprocessing with small spelling fixes ---
def preprocess_text(text):
    text = clean_text(text.lower())

    # Normalize common patterns
    replacements = {
        r"\bi feeling\b": "i am feeling",
        r"\blit bit\b": "little bit",
        r"\blit\b": "little",
        r"\bim\b": "i am",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    tokens = re.findall(r"\b\w+\b", text)
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])

# --- Load CSV ---
csv_files = [f for f in os.listdir() if f.endswith('.csv')]
if not csv_files:
    print("No CSV found in current directory.")
    sys.exit(1)

df = pd.read_csv(csv_files[0], encoding='utf-8', dtype=str, keep_default_na=False)
text_column = 'statement' if 'statement' in df.columns else df.columns[0]
df[text_column] = df[text_column].astype(str).apply(clean_text)
df['processed_text'] = df[text_column].apply(preprocess_text)
print(f"Loaded CSV: {csv_files[0]} shape={df.shape}")

# --- Load intents ---
if not os.path.exists('intents.json'):
    print("intents.json not found in current directory.")
    sys.exit(1)

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

for intent in intents.get('intents', []):
    intent['patterns'] = [clean_text(p) for p in intent.get('patterns', [])]
    intent['responses'] = [clean_text(r) for r in intent.get('responses', [])]

intent_tags = [it['tag'] for it in intents['intents']]
intent_to_patterns = {it['tag']: it.get('patterns', []) for it in intents['intents']}

# --- Compute mean embeddings for intents ---
intent_mean_embeddings = {}
for tag, patterns in intent_to_patterns.items():
    if not patterns:
        patterns = [tag]
    processed_patterns = [preprocess_text(p) for p in patterns]
    emb = embedder.encode(processed_patterns, convert_to_tensor=True)
    intent_mean_embeddings[tag] = emb.mean(dim=0)

# --- Encode statements from CSV ---
statement_embeddings = embedder.encode(df['processed_text'].tolist(), convert_to_tensor=True)
all_tags = list(intent_mean_embeddings.keys())
intent_tensor_list = [intent_mean_embeddings[t] for t in all_tags]
intent_tensor_stack = torch.stack(intent_tensor_list)

# --- Assign tags to CSV statements ---
cos_sim = util.pytorch_cos_sim(statement_embeddings, intent_tensor_stack).cpu().numpy()
assigned_tags = [all_tags[int(np.argmax(cos_sim[i]))] for i in range(cos_sim.shape[0])]
assigned_scores = [float(np.max(cos_sim[i])) for i in range(cos_sim.shape[0])]
df['assigned_tag'] = assigned_tags
df['assigned_score'] = assigned_scores

FALLBACK_TAG = 'fallback'
SIM_THRESHOLD = 0.35  # lowered for better matching
df['final_tag'] = [t if s >= SIM_THRESHOLD else FALLBACK_TAG for t, s in zip(assigned_tags, assigned_scores)]

if FALLBACK_TAG not in [t['tag'] for t in intents['intents']]:
    intents['intents'].append({
        "tag": FALLBACK_TAG,
        "patterns": ["", "random text", "What?", "I don't know"],
        "responses": [
            "I'm here to support you. Please type something related to how you feel.",
            "I'm sorry, I didn't understand that. Can you try asking differently?"
        ]
    })

# --- Prepare training data ---
X = embedder.encode(df['processed_text'].tolist())
unique_tags = sorted(list(set(df['final_tag'].tolist())))
tag_to_idx = {t: i for i, t in enumerate(unique_tags)}
idx_to_tag = {i: t for t, i in tag_to_idx.items()}
y_idx = np.array([tag_to_idx[t] for t in df['final_tag']])
y = to_categorical(y_idx, num_classes=len(unique_tags))

MODEL_FILENAME = 'intent_model.h5'
LABELMAP_FILENAME = 'label_map.json'

if os.path.exists(MODEL_FILENAME) and os.path.exists(LABELMAP_FILENAME):
    model = load_model(MODEL_FILENAME)
    with open(LABELMAP_FILENAME, 'r', encoding='utf-8') as f:
        idx_to_tag = json.load(f)
    idx_to_tag = {int(k): v for k, v in idx_to_tag.items()}
    tag_to_idx = {v: int(k) for k, v in idx_to_tag.items()}
else:
    model = Sequential([
        Dense(256, input_shape=(X.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(unique_tags), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=40, batch_size=16, verbose=1)
    model.save(MODEL_FILENAME)
    with open(LABELMAP_FILENAME, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in idx_to_tag.items()}, f, ensure_ascii=False, indent=2)

# --- Crisis detection keywords ---
crisis_keywords = ["suicide", "kill myself", "end it all", "die", "worthless", "end my life", "i want to die"]

def check_crisis(text):
    text = text.lower()
    return any(kw in text for kw in crisis_keywords)

# --- Sentiment analysis ---
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] <= -0.5: return "negative"
    if scores['compound'] >= 0.5: return "positive"
    return "neutral"

# --- Logging ---
if not os.path.exists("logs"):
    os.mkdir("logs")

def log_conversation(user_input, bot_response):
    with open("logs/conversation_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | User: {user_input} | Bot: {bot_response}\n")

# --- Conversation history ---
conversation_history = []

def get_intent_emb(emb_tensor):
    emb_array = emb_tensor.cpu().numpy()
    if emb_array.ndim == 1:
        emb_array = emb_array.reshape(1, -1)
    probs = model.predict(emb_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx_to_tag[idx], float(probs[idx])

def get_dataset_response_emb(emb_tensor):
    sims = util.pytorch_cos_sim(emb_tensor, statement_embeddings).cpu().numpy()[0]
    best_idx = int(np.argmax(sims))
    return df.iloc[best_idx][text_column], float(sims[best_idx])

# --- Weighted context embedding ---
def get_context_embedding(history, decay=0.8):
    emb_list = []
    weight = 1.0
    for msg in reversed(history[-10:]):
        emb = embedder.encode(preprocess_text(msg), convert_to_tensor=True)
        emb_list.append(emb * weight)
        weight *= decay
    return torch.stack(emb_list).mean(dim=0)

# --- Chatbot response (updated logic) ---
def chatbot_response(user_input):
    global conversation_history
    conversation_history.append(user_input)
    processed_input = preprocess_text(user_input)
    emb = embedder.encode([processed_input], convert_to_tensor=True)

    # Crisis detection first
    if check_crisis(user_input):
        response = ("⚠️ I'm really concerned about your safety. Call a helpline:\n"
                    "- Emergency (India): 112\n- KIRAN: 1800-599-0019\n- AASRA: +91-98204-66726")
        log_conversation(user_input, response)
        return response

    context_emb = get_context_embedding(conversation_history)
    tag, conf = get_intent_emb(context_emb)

    dataset_resp, ds_score = get_dataset_response_emb(emb)

    # --- Updated decision logic ---
    if conf < SIM_THRESHOLD:
        if ds_score >= 0.55:
            # Trust dataset if similarity is high
            intent_resp = dataset_resp
            tag = "dataset_match"
            conf = ds_score
        else:
            tag = FALLBACK_TAG
            intent_resp = random.choice(
                next((it['responses'] for it in intents['intents'] if it['tag'] == tag),
                     ["I'm here to support you."])
            )
    else:
        intent_resp = random.choice(
            next((it['responses'] for it in intents['intents'] if it['tag'] == tag),
                 ["I'm here to support you."])
        )

    # Add sentiment-based flavor
    sentiment = analyze_sentiment(user_input)
    if sentiment == "negative":
        intent_resp = "😔 I hear you. That sounds really tough. " + intent_resp
    elif sentiment == "positive":
        intent_resp = "😊 I'm glad you're feeling positive! " + intent_resp

    # Trim history
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    final_response = f"{intent_resp}"
    log_conversation(user_input, final_response)
    time.sleep(random.uniform(0.5, 1.5))
    return final_response

# --- Main loop ---
print("\nMental Health Chatbot: Hello! I'm here to provide support. Type 'bye' to exit.")
while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nMental Health Chatbot: Take care!")
        break
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Mental Health Chatbot: Take care!")
        break
    if not user_input:
        print("Mental Health Chatbot: Please type something.")
        continue

    print("Mental Health Chatbot:", chatbot_response(user_input))
