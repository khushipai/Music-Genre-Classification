import numpy as np
import pandas as pd
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import uniform, randint
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
import streamlit as st
import pickle

from pprint import pprint
import random
import librosa, IPython
import librosa.display as lplt
seed = 12
np.random.seed(seed)

df =pd.read_csv("C:\\Users\\khush\\OneDrive\\Desktop\\RNSIT\\music-genre-classification\\Data\\features_3_sec.csv")

# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x

# shuffle samples
df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# remove irrelevant columns
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_X = df_shuffle

# split into train dev and test
X_train, X_test, y_train, y_test= skms.train_test_split(df_X, df_y, train_size=0.7, random_state=seed, stratify=df_y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

import joblib
joblib.dump(classifier,'classifier')
model = joblib.load('classifier')


st.set_page_config(page_title="Music Genre Classification", layout="wide")
title = "Music Genre Classification"
st.title(title)
filename = st.file_uploader("Upload File",type=".wav")

if filename is not None:
    st.write("filename") 
    try:
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=57)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    except Exception as e:
            print('Got an exception:')
#filename="C:\\Users\\khush\\OneDrive\\Desktop\\RNSIT\\music-genre-classification\\Test\\rocktest2.wav"


#print(mfccs_scaled_features)

#print(mfccs_scaled_features)
#print(mfccs_scaled_features.shape)


if st.button('Predict'):
    predicted_label=model.predict(mfccs_scaled_features)
    st.write("The predicted genre is: {}".format(predicted_label))