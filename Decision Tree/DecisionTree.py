from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = load_breast_cancer()
X = data.data
y = data.target

#Decision Tree
x_train,x_test,y_train,y_test = train_test_split(X,y, random_state=42)
train_accuracy = []
test_accuracy = []
depths = range(1,25)
for depth in depths:
    Decision = DecisionTreeClassifier(max_depth=depth, random_state=0)
    Decision.fit(x_train,y_train)
    train_accuracy.append(Decision.score(x_train,y_train))
    test_accuracy.append(Decision.score(x_test,y_test))
plt.plot(depths,train_accuracy,label="Train Accuracy")
plt.plot(depths, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Depths")
plt.legend()
plt.show()

predict = Decision.predict(x_test)
confusion = confusion_matrix(y_test, predict)
sns.heatmap(confusion, annot=True)
plt.show()

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train,y_train)
print("Accuracy of testing set is: {:.2f}".format(tree.score(x_test,y_test)))   #0.96
print("Accuracy of training set is: {:.2f}".format(tree.score(x_train,y_train))) #0.97



