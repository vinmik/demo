from flask import Flask, request, jsonify
from ex5 import preprocess_text, get_most_appropriate_emoji

app = Flask(__name__)

@app.route('/get_emoji_code', methods=['POST'])
def get_emoji_code():
    sentence = request.json['sentence']
    input_sentence_preprocessed = preprocess_text(sentence)

    most_appropriate_emoji,emoji_code = get_most_appropriate_emoji(input_sentence_preprocessed)
    emoji_code = emoji_code.replace('U+', '&#x000', 1)+';'

    response = {
        'emoji': most_appropriate_emoji,
        'emoji_code' : emoji_code
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
