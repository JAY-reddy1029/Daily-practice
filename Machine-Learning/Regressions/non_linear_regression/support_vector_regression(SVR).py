import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\pvjay\nit\nit_practice.py\ML\regressions\non_linear_regression\emp_sal.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly',degree = 4,gamma="auto")
regressor.fit(X, y)

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=4)
regressor_knn.fit(X,y)

from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(splitter="random",criterion='absolute_error')
regressor_dt.fit(X,y)


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(random_state=0, n_estimators = 50)
reg_rf.fit(X,y)

# Predicting a new result

#y_pred_knn = regressor_knn.predict([[6.5]])

y_pred_svr = regressor.predict([[6.5]])
print(y_pred_svr)

#y_pred_dtr = regressor_dt.predict([[6.5]])

#y_pred_rf = reg_rf.predict([[6.5]])

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#if you check the output that is svr model & its predicting the each of the real observation
#red points are real observation point & blue lines are predicted line & now you can say svr is fitted much better curve on the dataset
#same hear if you check the ceo actual observation point but you will find as still we can improve the graph and lets see how can we do that in svr
#in this case ceo is outlier hear becuase ceo is quite far from our observation, thats ok

#what exactly we are doing hear to check the what exactly employees have 6.5yrs experience predict salary


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#great curve you got isn't it , same dataset you worked polynomial regressor & svm regressor 































