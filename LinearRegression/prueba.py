import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import datasets 
import matplotlib.pyplot as plt 
from scratch import linearregression

def mse(predicciones, y_real):
    mse=np.mean((predicciones-y_real)**2)
    return mse

X, y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)



reg=linearregression()
reg.fit(X_train,y_train)
prediccion=reg.predict(X_test)
scratch=mse(prediccion,y_test)

print(f"El error con el modelo implementado desde cero es {round(scratch,10)}")

model=LinearRegression()
model.fit(X_train,y_train)
prediccion_sck=model.predict(X_test)
scikit=mse(prediccion_sck,y_test)

print(f"El error con el modelo de scikit learn es {round(scikit,10)}")

print("diferencia de errores {}".format(scikit-scratch))

fig=plt.figure(figsize=(8,6))
x=np.linspace(-3,3,100)
y_scratch=(reg.weights)*x+reg.bias
y_scikit=(model.coef_)*x+model.intercept_
# plt.plot(x, y_scratch, color="blue")
plt.plot(x, y_scikit,  color="red")
plt.xlabel(f"{reg.weights}*x+{reg.bias}")
plt.scatter(X[:,0],y,color="b",marker="o",s=30)
plt.show()