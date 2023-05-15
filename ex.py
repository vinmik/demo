import json
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open('emojis.json',encoding ='utf-8') as json_file:
    emojis_data = json.load(json_file)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

input_sentence = "I am not feeling good today"
input_sentence_preprocessed = preprocess_text(input_sentence)

keywords = []
emojis = []
for category in emojis_data:
    for emoji in emojis_data[category]:
        for keyword in emoji['keywords']:
            keywords.append(preprocess_text(keyword))
            emojis.append(emoji['emoji'])

model_bert = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embedding = model_bert.encode([input_sentence_preprocessed])[0]
keyword_embeddings_bert = model_bert.encode(keywords)

similarity_scores = cosine_similarity([sentence_embedding], keyword_embeddings_bert)[0]

most_similar_index = similarity_scores.argmax()
most_similar_keyword = keywords[most_similar_index]
most_similar_emoji = emojis[most_similar_index]

print(f"Input Sentence: {input_sentence}")
print(f"Most Appropriate Emoji: {most_similar_emoji} - {most_similar_keyword}")
