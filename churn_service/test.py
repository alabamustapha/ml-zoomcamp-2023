import requests
import json

url = "http://172.26.247.137:9696/predict"

payload = json.dumps({
  "customerid": "0111-klbqg",
  "gender": "male",
  "seniorcitizen": 1,
  "partner": "yes",
  "dependents": "yes",
  "tenure": 32,
  "phoneservice": "yes",
  "multiplelines": "no",
  "internetservice": "fiber_optic",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "yes",
  "streamingmovies": "yes",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "mailed_check",
  "monthlycharges": 93.95,
  "totalcharges": 2861.45
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
