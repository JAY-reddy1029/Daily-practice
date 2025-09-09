import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\pvjay\nit\nit_practice.py\ML\regressions\non_linear_regression\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# svm model(Kernel,degree,gamma,C)
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree = 4, gamma = 'auto', C=10 )
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

# knn model (n-neighbors,weight,leaf_size,P)
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5,weights='distance', leaf_size=30, p=2)
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)


#decission tree algorithm
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X,y)

dt_reg_pred = dt_reg.predict([[6.5]])
dt_reg_pred

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=27,random_state=0)
rf_reg.fit(X,y)

rf_reg_pred = rf_reg.predict([[6.5]])
rf_reg_pred

# Visualising 
plt.scatter(X, y, color = 'red')
plt.plot(X,rf_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,rf_reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


























