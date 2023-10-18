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
