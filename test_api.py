import requests

url = 'http://127.0.0.1:5000/predict'
data = {'text': 'I love this project!'}

response = requests.post(url, json=data)
print(response.json())
