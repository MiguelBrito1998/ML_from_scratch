#Perceptron from scratch
import numpy as np

def unit_step_function(x):
    return np.where(x>0,1,0)


class Perceptron:
    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.lr=learning_rate
        self.n_iters=n_iters
        self.activation_func=unit_step_function
        self.weight=None
        self.bias=None

    def fit(self,X,y):
        n_samples,n_features=X.shape 
        #inicializar parametros
        self.weights=np.zeros(n_features)
        self.bias=0 

        y_=np.where(y>0,1,0)

        #learn weights 
        for _ in range(self.n_iters):
            for idx , x_i in enumerate(X):
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predicted=self.activation_func(linear_output)

                #Update rule 
                update=self.lr*(y_[idx]-y_predicted)
                self.weights+=update*x_i
                self.bias+=update

    def predict(self,X):
        y_predicted=np.dot(X,self.weights)+self.bias
        return y_predicted

