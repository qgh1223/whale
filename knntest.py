import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris_x=load_iris().data
iris_y=load_iris().target
knn=KNeighborsClassifier()
knn.fit(iris_x,iris_y)
print(knn.score(iris_x,iris_y))
distance,neighbor=knn.kneighbors(iris_x)
print(distance)
print(neighbor)
