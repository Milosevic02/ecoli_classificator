import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ecoli.csv',header=None,sep= '\s+')
col_names = ["sequence_name","mcg","gvh","lip","chg","aac","alm1","alm2","site"]
data.columns = col_names

X = data.iloc[:,1:8]
Y = data['site']

