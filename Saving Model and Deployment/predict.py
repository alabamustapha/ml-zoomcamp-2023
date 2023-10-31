
import pickle 

input_model = "model_C=1.0.bin"

with open(input_model, 'rb') as f_in:
    dv, model  = pickle.load(f_in)


customer_dict = {'customerid': '0111-klbqg',
 'gender': 'male',
 'seniorcitizen': 1,
 'partner': 'yes',
 'dependents': 'yes',
 'tenure': 32,
 'phoneservice': 'yes',
 'multiplelines': 'no',
 'internetservice': 'fiber_optic',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'yes',
 'streamingmovies': 'yes',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'mailed_check',
 'monthlycharges': 93.95,
 'totalcharges': 2861.45}


X = dv.transform(customer_dict)
predict_proba = model.predict_proba(X)[0, 1]

print(f"Probability of churning: {predict_proba}")
