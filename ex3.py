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

def main():
    input_sentence = "i am feeling hungry today"
    input_sentence_preprocessed = preprocess_text(input_sentence)

    keywords = []
    emojis = []
    for category in emojis_data:
        for emoji in emojis_data[category]:
            for keyword in emoji['keywords']:
                keywords.append(preprocess_text(keyword))
                emojis.append(emoji['emoji'])

    model_distilbert = SentenceTransformer('distilbert-base-nli-mean-tokens')

    try:
        embeddings_data = np.load('embeddings.npz')
        keyword_embeddings_distilbert = embeddings_data['embeddings']
    except FileNotFoundError:
        keyword_embeddings_distilbert = model_distilbert.encode(keywords)
        np.savez_compressed('embeddings.npz', embeddings=keyword_embeddings_distilbert)

    sentence_embedding = model_distilbert.encode([input_sentence_preprocessed])[0]
    similarity_scores = cosine_similarity([sentence_embedding], keyword_embeddings_distilbert)[0]

    most_similar_index = similarity_scores.argmax()
    most_similar_keyword = keywords[most_similar_index]
    most_similar_emoji = emojis[most_similar_index]

    print(f"Input Sentence: {input_sentence}")
    print(f"Most Appropriate Emoji: {most_similar_emoji} - {most_similar_keyword}")
    print(f"Caption: {input_sentence} {most_similar_emoji}")


main()
