import joblib
import sklearn.datasets

from util.datasets import load_dataset

"""
https://github.com/scikit-learn/scikit-learn/issues/19376

from sklearn.datasets import get_data_home
print(get_data_home())

"""


def olivetti():
    X,y = sklearn.datasets.fetch_olivetti_faces(return_X_y=True)
    print(X.shape)

def olivetti2():
    X,y = load_dataset('olivetti_faces_16')
    print(X.shape)

olivetti2()