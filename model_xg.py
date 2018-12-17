# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:32:21 2018

@author: Maddy
"""


# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
data = pd.read_csv('claim_API.csv')
data = data.drop(['Unnamed: 0'], axis=1)
data.head(5)
data.describe()
data = pd.get_dummies(data)

data.iloc[:,5:].head(5)

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(data['FraudFound_P'])
# Remove the labels from the features
# axis 1 refers to the columns
features= data.drop('FraudFound_P', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

import xgboost as xgb
from sklearn import metrics
model = xgb.XGBClassifier(n_estimators=500, max_depth=5,
                        objective='binary:logistic', random_state=42)
model.fit(X_train,y_train)

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))
auc(model, X_train, data)

from sklearn.externals import joblib
joblib.dump(model, 'model.pkl')


xg = joblib.load('model.pkl')
xg.predict(X_test)