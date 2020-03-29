from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_test, x_train,y_test,y_train = train_test_split(X,y, stratify=cancer.target, random_state=64)
train_acc = []
test_acc = []

neighbors = range(1,25)
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

#Try with 6-Neighbors
kn = KNeighborsClassifier(n_neighbors=6)
kn.fit(x_train,y_train)
print("The train accuracy for 8N is: {:.2f}".format(kn.score(x_train,y_train)))  
print("The test accuracy for 8N is: {:.2f}".format(kn.score(x_test, y_test)))