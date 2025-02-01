import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
import matplotlib.pyplot as plt 
from scratch import Logistic_Regression

bc=datasets.load_breast_cancer()
X,y=bc.data,bc.target 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1234)

clr=Logistic_Regression(lr=0.01)
clr.fit(X_train,y_train)
y_pred=clr.predict(X_test)

#Precision
def accuracy(y_pred,y_test):
    return np.sum((y_pred==y_test)/len(y_test))

acc=accuracy(y_pred,y_test)
print(acc)


