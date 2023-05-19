import requests

url = 'http://localhost:5000/get_emoji_code'
data = {
    'sentence': 'I am so sad that I feel happy'
}

response = requests.post(url, json=data)
response_data = response.json()

if 'emoji' in response_data and 'emoji_code' in response_data:
    emoji = response_data['emoji']
    emoji_code = response_data['emoji_code']
    print(f"Emoji: {emoji}")
    print(f"Emoji Code: {emoji_code}")
else:
    print("Error: Emoji or emoji code not found in the response.")
