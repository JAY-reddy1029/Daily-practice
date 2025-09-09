import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv(r"C:\Users\pvjay\nit\nit_practice.py\ML\Data (1).csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer # to fill missing data
imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder # to convert categorical data to numarical data.

labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0])

x[:,0] = labelencoder_x.fit_transform(x[:,0]) # for x data

labelencoder_y = LabelEncoder()  # to convert categorical data to numarical data.
y = labelencoder_y.fit_transform(y) # for y data

from sklearn.model_selection import train_test_split # to split the data to train and test 80% and 20%
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, test_size=0.2 , random_state=0) # write any 1 either test or train.(use random_state=0 to get same split everytime)

#regression
#(simple linear reg alg => y=mx+c)


















