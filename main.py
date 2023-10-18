import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ecoli.csv',header=None,sep= '\s+')
col_names = ["sequence_name","mcg","gvh","lip","chg","aac","alm1","alm2","site"]
data.columns = col_names

X = data.iloc[:,1:8]
y = data['site']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
N,D = X_train.shape

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,input_shape = (7,),activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,activation = 'relu'),
    tf.keras.layers.Dense(8,activation = 'softmax')
])

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

def encode_labels(labels):
    label_mapping = {"cp": 0, "im": 1, "imS": 2, "imL": 3, "imU": 4, "om": 5, "omL": 6, "pp": 7}
    encoded_labels = [label_mapping[label] for label in labels]
    return encoded_labels

y_train_numeric = encode_labels(y_train)
y_test_numeric = encode_labels(y_test)

