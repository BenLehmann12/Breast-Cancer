from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


data = load_breast_cancer()
X = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,y, random_state=42)

regression = LogisticRegression(C=2)
regression.fit(x_train,y_train)
print("The testing accuracy is: {:.4f}".format(regression.score(x_test,y_test)))  #0.9580
print("The training accuracy is: {:.4f}".format(regression.score(x_train,y_train))) #0.9577

predict = regression.predict(x_test)
confusion = confusion_matrix(y_test, predict)
sns.heatmap(confusion, annot=True)
plt.show()

