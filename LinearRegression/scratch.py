#Linear regression from scratch
import numpy as np
class linearregression:
    def __init__(self, lr=0.001,n_iters=10000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None


    def fit(self, X, y):
        n_datos,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            y_pred=np.dot(X,self.weights)+self.bias   # Se usa X ya que necesitamos una prediccion por cada muestra
            dw=(1/n_datos)*np.dot(X.T,(y_pred-y))       # Se usa la X.T ya que necesitamos un dw por cada feature
            db=(1/n_datos)*np.sum(y_pred-y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db
        print(dw)
        print(db)
        print(y_pred)


    def predict(self, X):
        y_pred=np.dot(X,self.weights)+self.bias
        return y_pred





