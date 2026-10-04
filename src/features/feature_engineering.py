import numpy as np
import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer

max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

# fetch data from the data/processed
train_data = pd.read_csv('data/processed/train_processed.csv')
test_data = pd.read_csv('data/processed/test_processed.csv')

# Apply Bag of Words (BoW) feature extraction technique to the processed data

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features= max_features)


# Wrap the numpy arrays in a Pandas Series to use the fillna method
X_train = pd.Series(X_train).fillna("")
X_test = pd.Series(X_test).fillna("")

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df['label'] = y_test

# store the data in the data/feature folder

data_path = os.path.join('data', 'featured')
os.makedirs(data_path, exist_ok=True)
train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)        
