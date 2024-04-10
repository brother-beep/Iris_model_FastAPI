import json
import requests

url = "http://127.0.0.1:8000/predict"

input_data_for_model = {
    'sepal_length': 5.9,
    'sepal_width':3.0,
    'petal_length':5.1,
    'petal_width':2.3
}

input_json = json.dumps(input_data_for_model)
response = requests.post(url,data=input_json)
print(response.text)