import numpy as np
from sklearn.datasets import make_regression

from glmnet import ElasticNet

if __name__ == "__main__":
    np.random.seed(488881)
    X, y = make_regression(n_samples=1000, random_state=561)
    m = ElasticNet()
    m.fit(X, y)
    p = m.predict(X[0].reshape((1, -1)), lamb=[20, 10])