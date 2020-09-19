# I created this script based on my learnings from the Machine Learning with Python course, part of the IBM data science centification in Coursera.
# The dataset I used in this script is from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv
# The dataset includes the CO2 emissions of cars along with car information such as engine size, cylinders, fuel consumption in city and highway.
# 80% of the data is randomly selected as the training set; and the rest as the test dataset. Model evaluation was performed using the test dataset.

# I tested 3 multivariate linear regression models to estimate the car CO2 emissions. 
# In Model 1: I used engine size and cylinders to estimate emissions;
# In Model 2: I used engine size, cylinders as well as the fuel consumption in city and highway to estimate emissions;
# In Model 3: I used same features as in Model 2 but this time used a 2nd degree polynomial fitting on engine size.
# To get a sense of accuracy of the models, I created scatter plots of predicted and actual emissions vs engine size on the test dataset.
# I also looked at mean absolute and squared errors, and R2 score to compare the model results quantitatively.
# While addition of fuel consumption improved the predictions largely, the polynomial fitting of the engine size did not result in significant improvements.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# download csv file with cars fuel consumption and emission data
!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv

# read data into a dataframe
df = pd.read_csv("FuelConsumption.csv")

# split data into train and test sets
train_inds = np.random.rand(len(df)) < 0.8   # creating a series of boolean variables with 80% Trues and 20% Falses
df_train = df[train_inds]  # use the randomly selected indices (Trues) to define the training set
df_test = df[~train_inds] # use the rest of the indices (Fales) to define the test set

LR = linear_model.LinearRegression() # create the linear regression object
#########################################################################
# Model 1:
#########################################################################
# Train the linear regression model using 'ENGINESIZE' and'CYLINDERS'
x_train = np.asanyarray(df_train[['ENGINESIZE','CYLINDERS']])
y_train = np.asanyarray(df_train[['CO2EMISSIONS']])
LR.fit (x_train, y_train)

# create a scatter plot to visualize the accuracy of the model
x_test = np.asanyarray(df_test[['ENGINESIZE','CYLINDERS']])
y_test = np.asanyarray(df_test[['CO2EMISSIONS']])
y_hat= LR.predict(x_test)  # predicted emissions for the test set

# actual and predicted emissions plotted vs engine size
plt.figure(figsize=(8,6))
plt.scatter(df_test.ENGINESIZE, df_test.CO2EMISSIONS,  color='blue')
plt.plot(x_test[:,0], y_hat, '+r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear regression model using engine size and cylinders")
plt.legend(['prediction','actual'],loc='lower right')

# evaluate the accuracy of the predictions using different metrics 
print("Mean Absolute Error (MAE): %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y_test) )

#########################################################################
# Model 2:
#########################################################################

# train the linear regression model this time using 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY'and'FUELCONSUMPTION_HWY'
x_train = np.asanyarray(df_train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(df_train[['CO2EMISSIONS']])
LR.fit (x_train, y_train)


# create a scatter plot to visualize the accuracy of the model
x_test = np.asanyarray(df_test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(df_test[['CO2EMISSIONS']])
y_hat= LR.predict(x_test)  # predicted emissions for the test set

# actual and predicted emissions plotted vs engine size
plt.figure(figsize=(8,6))
plt.scatter(df_test.ENGINESIZE, df_test.CO2EMISSIONS,  color='blue')
plt.plot(x_test[:,0], y_hat, '+r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear regression model using engine size, cylinders and fuel consumption")
plt.legend(['prediction','actual'],loc='lower right')
plt.savefig('LR2.png')

# evaluate the accuracy of the predictions using different metrics 
print("Mean Absolute Error (MAE): %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y_test) )

#########################################################################
# Model 3
#########################################################################

# define the training dataset, using a second degree polynomial feature for 'ENGINESIZE'
x_train_1 = np.asanyarray(df_train[['ENGINESIZE']]) 
x_train_2 = np.asanyarray(df_train[['CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train_1)
x_train = np.concatenate((x_train_poly2, x_train_2),axis=1)
y_train = np.asanyarray(df_train[['CO2EMISSIONS']])

# fit the linear regression model based on training dataset
LR.fit(x_train, y_train)

# define the test dataset
x_test_1 = np.asanyarray(df_test[['ENGINESIZE']])
x_test_2 = np.asanyarray(df_test[['CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x_test_poly2 = poly2.fit_transform(x_test_1)
x_test = np.concatenate((x_test_poly2, x_test_2),axis=1)
y_test = np.asanyarray(df_test[['CO2EMISSIONS']])
# calculate predicted emissions for the test dataset
y_hat= LR.predict(x_test)

# plot actual and predicted emissions vs engine size
plt.figure(figsize=(8,6))
plt.scatter(df_test.ENGINESIZE, df_test.CO2EMISSIONS,  color='blue')
plt.plot(x_test[:,1], y_hat, '+r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.legend(['prediction','actual'],loc='lower right')
plt.title("Linear regression model using cylinders, fuel consumption \n and 2nd degree polynomial fitting of engine size")

# evaluate the accuracy of the predictions using different metrics 
print("Mean Absolute Error (MAE): %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y_test) )
