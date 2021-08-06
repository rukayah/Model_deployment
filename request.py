import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sl':5.0, 'sw':3.6, 'pl':1.4, 'pw':0.2})

print(r.json())