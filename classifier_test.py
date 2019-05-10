from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

iris = datasets.load_iris()

data = pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

clf = pickle.load(open('classifier.pkl', 'rb'))
y_pred=clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

species_idx = clf.predict([[3, 5, 4, 2],[3, 10, 14, 2]])
print(species_idx)
