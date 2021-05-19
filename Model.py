import numpy as np
import pandas as pd
import pickle
from grams import grams

from nltk.util import everygrams

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train['News']
y = train['Target']

X_test = test['News']
y_test = test['Target']

def grams(text):
    #Character grams
    for i in list(everygrams(''.join([c for c in text if c is not ' ']), min_len=1, max_len=4)):
        yield i
    #Word grams
    for i in list(everygrams(text.split(' '), min_len=1, max_len=1)):
        yield i

XGB = XGBClassifier(colsample_bylevel = 0.1, colsample_bynode = 0.775, colsample_bytree = 0.325, learning_rate = 0.3,
max_delta_step = 1, min_child_weight = 3, n_estimators = 400, reg_alpha = 5, reg_lambda = 1, subsample = 1.0
)

vector = TfidfVectorizer(analyzer=grams)

model = Pipeline([('vector', vector),
                  ('classifier', XGB),
                ])
model.fit(X, y)

pickle.dump(model, open("ml_model.sav", "wb"))
