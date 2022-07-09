#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#read file and check data correlation heatmap
data = pd.read_csv('lifeexpectancydata.csv')
sns.heatmap(data.corr())
plt.show()
#Trying to identify linearly paired data
sns.scatterplot(data=data, x="GDP",y="Life expectancy")
plt.show()
sns.scatterplot(data=data, x="Schooling",y="Life expectancy")
plt.show()
sns.scatterplot(data=data, x="Income composition of resources",y="Life expectancy")
plt.show()
sns.scatterplot(data=data, x="Income composition of resources",y="Schooling")
plt.show()

data.loc[data['Country'] == "India"]
#India Data Correlation Heatmap
indata = data.loc[data['Country'] == "India"]
sns.heatmap(indata.corr())
plt.show()
sns.scatterplot(data=indata, x="GDP",y="Life expectancy")
plt.show()
sns.scatterplot(data=indata, x="Year",y="Life expectancy")
plt.show()
###Modeling Year vs Life expectancy
X = indata.iloc[:, 1:2].values  #Year
#y = dataset.iloc[:, 3].values
Y=indata['Life expectancy'].values

##Run Regression algo
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Visualization
sns.scatterplot(data=indata, x="GDP", y="Life expectancy")
plt.show()
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#intercept value: c
print("Intercept value: ",regressor.intercept_)
#coef as m
print("Coefficient value: ",regressor.coef_)
#predicting for 2021 year
pred_2021 = regressor.coef_ * 2021 + regressor.intercept_
print(" Life Expectancy for the year 2021 is: ", pred_2021)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Prediction for X Test Data: ",y_pred)
import numpy as np
from sklearn import metrics
explained_variance=metrics.explained_variance_score(y_test, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
mse=metrics.mean_squared_error(y_test, y_pred)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
r2=metrics.r2_score(y_test, y_pred)
print('Explained_variance: ', round(explained_variance,2))
print('Mean_Squared_Log_Error: ', round(mean_squared_log_error,2))
print('R-squared: ', round(r2,4))
print('Mean Absolute Error(MAE): ', round(mean_absolute_error,2))
print('Mean Squared Error (MSE): ', round(mse,2))
print('Root Mean Squared Error (RMSE): ', round(np.sqrt(mse),2))

#########  Multiple Regression

###Modeling Year vs Life expectancy
X = indata.iloc[:, [1,4,5]].values  #Year
#y = dataset.iloc[:, 3].values
Y=indata['Life expectancy'].values



##Run Regression algo
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Visualization
sns.scatterplot(data=indata, x="Year", y="Life expectancy")
plt.show()
sns.scatterplot(data=indata, x="Adult Mortality", y="Life expectancy")
plt.show()
sns.scatterplot(data=indata, x="infant deaths", y="Life expectancy")
plt.show()
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#intercept value: c
print("Intercept value: ",regressor.intercept_)
#coef as m
print("Coefficient value: ",regressor.coef_)
#predicting for 2021 year
pred_2021 = regressor.coef_[0] * 2021 + regressor.coef_[1] * 291 + regressor.coef_[2] * 91+ regressor.intercept_
print(" Life Expectancy for the year 2021 is: ", pred_2021)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Prediction for X Test Data: ",y_pred)
import numpy as np
from sklearn import metrics
explained_variance=metrics.explained_variance_score(y_test, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
mse=metrics.mean_squared_error(y_test, y_pred)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
r2=metrics.r2_score(y_test, y_pred)
print('Explained_variance: ', round(explained_variance,2))
print('Mean_Squared_Log_Error: ', round(mean_squared_log_error,2))
print('R-squared: ', round(r2,4))
print('Mean Absolute Error(MAE): ', round(mean_absolute_error,2))
print('Mean Squared Error (MSE): ', round(mse,2))
print('Root Mean Squared Error (RMSE): ', round(np.sqrt(mse),2))

from statsmodels.api import OLS
import statsmodels.api as sm
#In our model, y will be dependent on 2 values: coefficienct
# and constant, so we need to add additional column in X for
#constant value
X = sm.add_constant(X)
summ = OLS(Y, X).fit().summary()
print("Summary of the dataset: \n",summ)

