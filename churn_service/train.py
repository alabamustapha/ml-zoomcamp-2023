import pickle

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score

# Parameters
C=1.0 
file_name = f"model_C={C}.bin"

# get dataset if missing
if not os.path.isfile("telco-churn.csv"):
    get_ipython().system('wget -O telco-churn.csv "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"')

# data preparation
df = pd.read_csv("telco-churn.csv")
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = df.dtypes[df.dtypes == 'object'].index
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


df.totalcharges = pd.to_numeric(df['totalcharges'], errors='coerce') #change to number and set invalid to NaN
df.totalcharges = df.totalcharges.fillna(0)
df['churn'] = df.churn.map({'yes': 1, 'no': 0})


# validation setup

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
y_train = df_train.churn
del df_train['churn']

df_val = df_val.reset_index(drop=True)
y_val = df_val.churn
del df_val['churn']

df_test = df_test.reset_index(drop=True)
y_test = df_test.churn
del df_test['churn']


numerical =  ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# training the model

dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]
churn_predictions = y_pred >= 0.5

train_auc = roc_auc_score(y_val, y_pred)
print(f"Training AUC: {train_auc}")

# full training
train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts) 
y_train = df_full_train.churn
test_dicts = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dicts)


model = LogisticRegression(C=C)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

full_train_auc = roc_auc_score(y_test, y_pred)

print(f"Full training AUC: {full_train_auc}")

# saving the model
with open(file_name, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model saved in: {file_name}")