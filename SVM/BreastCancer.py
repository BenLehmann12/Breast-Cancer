import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=0)

#SVC
support = SVC()
support.fit(x_train,y_train)
print("The testing accuracy is: {:.2f}".format(support.score(x_test,y_test)))  #0.63
print("The training accuracy is: {:.2f}".format(support.score(x_train,y_train))) #1.00
prediction = support.predict(x_test)
matrix = confusion_matrix(y_test, prediction)
sns.heatmap(matrix, annot=True)
plt.show()

#Scaled SVC
scaled = MinMaxScaler()
scaled_x_train = scaled.fit_transform(x_train)
scaled_x_test = scaled.fit_transform(x_test)
vector = SVC()
vector.fit(scaled_x_train,y_train)
print("The testing accuracy is: {:.2f}".format(vector.score(scaled_x_test, y_test)))
print("The training accuracy is: {:.2f}".format(vector.score(scaled_x_train, y_train)))
scaled_prediction = vector.predict(scaled_x_test)
scaled_matrix = confusion_matrix(y_test, scaled_prediction)
sns.heatmap(scaled_matrix, annot=True)
plt.show()