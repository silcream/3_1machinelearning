from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
# print(dataset.data) #data for the various columns of the dataset
# print(dataset.feature_names) #name of each column
# print(dataset.DESCR) # description of each feature
# print(dataset.target) # The prices of houses
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target #add the prices of the houses to the DataFrame
# display(df.head()) # show first 5 rows
# display(df.tail()) # show last 5 rows
df.info() #check the data type of each field
print(df.isnull().sum()) #check to see if there are any missing values

corr = df.corr() # computes the pairwise correlation of columns
print(corr)
#---get the top 3 features that has the highest correlation---
print(df.corr().abs().nlargest(3, 'MEDV').index)
#---print the top 3 correlation values---
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])

plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')

plt.scatter(df['RM'], df['MEDV'], marker='o')
plt.xlabel('RM')
plt.ylabel('MEDV')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['LSTAT'], df['RM'], df['MEDV'], c='b')
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
plt.show()

# linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM']) 
#combination of the LSTAT and RM features 
Y = df['MEDV'] #contain the MEDV label
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3,random_state=5) 
#split the dataset into 70 percent for training and 30 percent
# print(x_train.shape)
# print(Y_train.shape)
# print(x_test.shape)
# print(Y_test.shape)
model = LinearRegression()
model.fit(x_train, Y_train) #perform linear regression
price_pred = model.predict(x_test) #perform predictions using R-Squared method
print('R-Squared: %.4f' % model.score(x_test,Y_test))

mse = mean_squared_error(Y_test, price_pred) #perform predictions using mse method
print(mse)
plt.scatter(Y_test, price_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")
print(model.intercept_) #bias
print(model.coef_) #weight
print(model.predict([[30,5]])) #ex. prediction for the house price when LSTAT is 30 and RM is 5

# # 3D regression hyperplane
# fig = plt.figure(figsize=(18,15))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x['LSTAT'], x['RM'], Y, c='b')
# ax.set_xlabel("LSTAT")
# ax.set_ylabel("RM")
# ax.set_zlabel("MEDV")
# #---create a meshgrid of all the values for LSTAT and RM---
# x_surf = np.arange(0, 40, 1) #---range for LSTAT---
# y_surf = np.arange(0, 10, 1) #--- range for RM---
# x_surf, y_surf = np.meshgrid(x_surf, y_surf) #두 개의 1-D 배열에서 사각형 그리드
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x, Y)
# #---calculate z(MEDC) based on the model---
# z = lambda x,y: (model.intercept_ + model.coef_[0] * x + model.coef_[1] * y)
# ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf), rstride=1, cstride=1, color='None', alpha = 0.4)
# plt.show()
