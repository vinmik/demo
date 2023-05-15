import json
import re
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open('emojis.json', encoding='utf-8') as json_file:
    emojis_data = json.load(json_file)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_embeddings(sentences, model):
    embeddings = model.encode(sentences)
    return embeddings

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    embeddings = np.load(file_path)
    return embeddings

input_sentence = "I am not feeling good today"
input_sentence_preprocessed = preprocess_text(input_sentence)

keywords = []
emojis = []
for category in emojis_data:
    for emoji in emojis_data[category]:
        for keyword in emoji['keywords']:
            preprocessed_keyword = preprocess_text(keyword)
            keywords.append(preprocessed_keyword)
            emojis.append(emoji['emoji'])

model_distilbert = SentenceTransformer('distilbert-base-nli-mean-tokens')

try:
    keyword_embeddings = load_embeddings('keyword_embeddings.npy')
except FileNotFoundError:
    keyword_embeddings = calculate_embeddings(keywords, model_distilbert)
    save_embeddings(keyword_embeddings, 'keyword_embeddings.npy')

input_embedding = calculate_embeddings([input_sentence_preprocessed], model_distilbert)[0]

similarity_scores = cosine_similarity([input_embedding], keyword_embeddings)[0]

most_similar_index = similarity_scores.argmax()
most_similar_keyword = keywords[most_similar_index]
most_similar_emoji = emojis[most_similar_index]

print(f"Input Sentence: {input_sentence}")
print(f"Most Appropriate Emoji: {most_similar_emoji} - {most_similar_keyword}")
