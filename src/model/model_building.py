import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier

n_estimators = yaml.safe_load(open('params.yaml','r'))['Model_building']['n_estimators']
learning_rate = yaml.safe_load(open('params.yaml','r'))['Model_building']['learning_rate']
#fetch data from the data/processed
train_data = pd.read_csv('data/processed/train_tfidf.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values


clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
clf.fit(X_train, y_train)


pickle.dump(clf, open('models/model.pkl', 'wb'))