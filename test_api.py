import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Swiggy 340"}
response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Raw response:", response.text)
