from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = load_breast_cancer()
X = data.data
y = data.target

x_test, x_train,y_test,y_train = train_test_split(X,y, stratify=data.target, random_state=64)
train_acc = []
test_acc = []

neighbors = range(1,35)
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_acc.append(knn.score(x_train,y_train))
    test_acc.append(knn.score(x_test,y_test))

plt.plot(neighbors, train_acc, label = "Training Accuracy")
plt.plot(neighbors, test_acc, label = "Testing Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Neighbors")
plt.legend()
plt.show()

