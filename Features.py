from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


data = load_breast_cancer()
X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=4)  #max_depth = more tree splits, more splits= more data
forest.fit(x_train,y_train)

#Confusion Matrix of Random Forest
predict = forest.predict(x_test)
confusion = confusion_matrix(y_test, predict)
sns.heatmap(confusion, annot=True)
plt.show()

print("Accuracy of Testing is: {:.2f}".format(forest.score(x_test,y_test)))
print("Accuracy of Training is: {:.2f}".format(forest.score(x_train,y_train)))

#Graphing the Random FOrest accuracy
train_acc = []
test_acc = []
depths = range(1,15)
for k in depths:
    tree = RandomForestClassifier(max_depth=k, random_state=42, n_estimators=100)
    tree.fit(x_train,y_train)
    train_acc.append(tree.score(x_train,y_train))
    test_acc.append(tree.score(x_test,y_test))
plt.plot(depths, train_acc, label="Train Accuracy")
plt.plot(depths, test_acc, label="Testing Accuuracy")
plt.ylabel("Accuracy")
plt.xlabel("Depth")
plt.show()