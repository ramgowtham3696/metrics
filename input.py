from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
print(X.shape)
print(y.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X, y)
y_pred=knn.predict(X)
from sklearn import metrics
print(metrics.accuracy_score(y,y_pred))