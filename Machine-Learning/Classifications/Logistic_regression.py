import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\pvjay\nit\nit_practice.py\ML\clasifications\Logistic_regression\logit classification.csv")

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='lbfgs')
classifier.fit(x_train,y_train)
#we have to fit the logistic regression model to our training set

# Predicting the Test set results
y_pred=classifier.predict(x_test)


#now we will use the confusion matrix to evalute

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)

bias=classifier.score(x_test,y_test)
print(bias)

variance=classifier.score(x_train,y_train)
print(variance)

#---------------future prediction----------------

dataset1=pd.read_csv(r"C:\Users\pvjay\nit\nit_practice.py\ML\clasifications\Logistic_regression\final1.csv")

d2=dataset1.copy()

dataset1=dataset1.iloc[:,[4,5]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
m=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()

d2['y_pred1']=classifier.predict(m)

d2.to_csv('final1.csv')













































