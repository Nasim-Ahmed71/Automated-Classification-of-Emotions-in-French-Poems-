import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrices
from selenium import webdriver
form sklearn.linear_model import SGDClassifier
from sklearn.linear_model import KNeighborsClassifier


path = "url of our dataset"
data = pd.read_csv(path)



X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X, Y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('sgdclassifier', SGDClassifier())])
print(clf.predict([[-0.8, -1]]))

