from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



cancer = load_breast_cancer()
cancer_frame = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'],['target']))
X = cancer_frame.drop(['target'], axis=1)
y = cancer_frame['target']


#DecisionTree find the best depth
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=35 )
train_accuracy = []
test_accuracy = []
depths = range(1,20)
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

scale = StandardScaler()
x_train_scale = scale.fit_transform(x_train)
x_test_scale = scale.transform(x_test)

#Scaled DecisionTree accuracy
scaledTree = DecisionTreeClassifier(criterion='entropy', random_state=50)
scaledTree.fit(x_train_scale, y_train)
y_pred = scaledTree.predict(x_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#Regular DecisionTree
tree = DecisionTreeClassifier(criterion='entropy', random_state=50)
tree.fit(x_train,y_train)
new_pred = tree.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)



