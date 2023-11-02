
from flask import Flask, request, jsonify
import pickle 



app = Flask('ping')
input_model = "model_C=1.0.bin"

with open(input_model, 'rb') as f_in:
    dv, model  = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()
    X = dv.transform(customer_dict)
    predict_proba = model.predict_proba(X)[0, 1]

    churn = predict_proba >= 0.5

    result = {
        'churn_probability': predict_proba
        'churn': churn
    }

    return jsonify(result)




if __name__ == '__main__':
    app.run(debug=True, port=9696, host='0.0.0.0')