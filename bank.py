import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')

x = df.iloc[:,3:13].values
y = df.iloc[:,13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoderx1 = LabelEncoder()
x[:,1] = labelencoderx1.fit_transform(x[:,1])
labelencoderx2 = LabelEncoder()
x[:,2] = labelencoderx2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.1,random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=6, init = 'uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init = 'uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init = 'uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
classifier.fit(xtrain,ytrain, batch_size=10, epochs=100)

ypred = classifier.predict(xtest)
ypred = (ypred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,ypred)