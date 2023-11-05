
from flask import Flask, request, jsonify
import pickle 


# start flask app
app = Flask('churn')
input_model = "model_C=1.0.bin"

# load save models
with open(input_model, 'rb') as f_in:
    dv, model  = pickle.load(f_in)

# route for post request
@app.route('/predict', methods=['POST'])
def predict():
   
    customer_dict = request.get_json()
    
    X = dv.transform(customer_dict)
    predict_proba = model.predict_proba(X)[0, 1]

    churn = predict_proba >= 0.5

    result = {
        "churn_probability": float(predict_proba),
        "churn": bool(churn)
    }

    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, port=9696, host='0.0.0.0')