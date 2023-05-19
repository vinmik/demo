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

def get_most_appropriate_emoji(input_sentence_preprocessed):
    keywords = []
    emoji_codes = []
    emojis = []
    for category in emojis_data:
        for emoji in emojis_data[category]:
            for keyword in emoji['keywords']:
                keywords.append(preprocess_text(keyword))
                emoji_codes.append(emoji['code'])
                emojis.append(emoji['emoji'])

    model_distilbert = SentenceTransformer('bert-base-nli-mean-tokens')

    try:
        embeddings_data = np.load('embeddings.npz')
        keyword_embeddings_distilbert = embeddings_data['embeddings']
    except FileNotFoundError:
        keyword_embeddings_distilbert = model_distilbert.encode(keywords)
        np.savez_compressed('embeddings.npz', embeddings=keyword_embeddings_distilbert)

    sentence_embedding = model_distilbert.encode([input_sentence_preprocessed])[0]
    similarity_scores = cosine_similarity([sentence_embedding], keyword_embeddings_distilbert)[0]

    most_similar_index = similarity_scores.argmax()
    most_similar_emoji = emojis[most_similar_index]
    most_similar_emoji_code = emoji_codes[most_similar_index]

    return most_similar_emoji, most_similar_emoji_code
