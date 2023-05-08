'''
This file is used to import the dataset, train a model and save it.
'''

import numpy as np
import pandas as pd

print('We begin by loading the iris dataset as an example')
from sklearn.datasets import load_iris
iris = load_iris()

print('Now, we store the feature matrix (x) and response vector (y)')
X = iris.data
y = iris.target

print('We proceed to check the shapes of X and y')
print ('X.shape', X.shape)
print ('y.shape', y.shape)

print('Lets examine first 5 rows of feature matrix')
print(pd.DataFrame(X, columns=iris.feature_names).head())

print('Lets also see y, the response vector')
print(y)

#import the class
from sklearn.neighbors import KNeighborsClassifier

print('Now we instantiate the model (with the default parameters)')
knn = KNeighborsClassifier()

print('We fit the model with data (occurs in-place)')
knn.fit(X,y)

print('We predict the response for a new observation')
print('knn.predict([[3,5,4,1]]) =', knn.predict([[3,5,4,1]]))

#Note: Say if the features we specify above is 5 instead of 4, it will miserably fail. This is 
# beacuse the model is trained with only 4 features like Sepal length, Sepal Width, Petal length, and Petal width.

#Similarly, if the features mentioned are not in the specified order, it will also create an error. It has to follow 
#the order as mentioned.

print('Now, we finish by exporting our model')

print('We import pickle')
import pickle

print('Now, we open a file. Its important to use binary mode')
knnPickle = open('model/knnpickle_file', 'wb') 
      
print('We give it source and destination')
pickle.dump(knn, knnPickle)  

print('We close the file')
knnPickle.close()