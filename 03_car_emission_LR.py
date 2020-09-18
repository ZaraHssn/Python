# I created this script based on my learnings from Machine Learning with Python (IBM).
# The dataset I used in this script is from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv
# The dataset includes the CO2 emissions of cars along with car information such as engine size, cylinders, fuel consumption in city and highway.

# Two multivariate linear regression models are implemented to estimate the car CO2 emissions. 
# In the first model two variables are used: engine size and cylinders, while in the second model fuel consumption in city and highway are also included.
# The improvements achieved by addition of the fuel consumption parameters are examined by comparing the mean square errors as well as 
# scatter plots of the predicted and actual emission rates for the test data set. 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# download csv file with cars fuel consumption and emission data
!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv

# read data into a dataframe
df = pd.read_csv("FuelConsumption.csv")

# split data into train and test sets
train_inds = np.random.rand(len(df)) < 0.8   # creating a series of boolean variables with 80% Trues and 20% Falses
df_train = df[train_inds]  # use the randomly selected indices (Trues) to define the training set
df_test = df[~train_inds] # use the rest of the indices (Fales) to define the test set

from sklearn import linear_model # import linear regression model
LR = linear_model.LinearRegression() # create the linear regression object


# Model 1:

# Train the linear regression model using 'ENGINESIZE' and'CYLINDERS'
x_train = np.asanyarray(df_train[['ENGINESIZE','CYLINDERS']])
y_train = np.asanyarray(df_train[['CO2EMISSIONS']])
LR.fit (x_train, y_train)

# predict emissions for the test set
y_hat= LR.predict(df_test[['ENGINESIZE','CYLINDERS']])

# calculate mean squared error and create a scatter plot to visualize the accuracy of the model
x_test = np.asanyarray(df_test[['ENGINESIZE','CYLINDERS']])
y_test = np.asanyarray(df_test[['CO2EMISSIONS']])
MSE_1 = np.mean((y_hat - y_test) ** 2)

# actual and predicted emissions plotted vs engine size
plt.scatter(df_test.ENGINESIZE, df_test.CO2EMISSIONS,  color='blue')
plt.plot(x_test[:,0], y_hat, '+r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear regression model using engine size and cylinders")
plt.legend(['prediction','actual'],loc='lower right')
plt.show()

# Model 2:

# train the linear regression model ths time using ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY'and'FUELCONSUMPTION_HWY'
x_train = np.asanyarray(df_train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(df_train[['CO2EMISSIONS']])
LR.fit (x_train, y_train)

# predict emissions for the test set
y_hat= LR.predict(df_test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])

# calculate mean squared error and create a scatter plot to visualize the accuracy of the model
x_test = np.asanyarray(df_test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(df_test[['CO2EMISSIONS']])
MSE_2 = np.mean((y_hat - y_test) ** 2)

# actual and predicted emissions plotted vs engine size
plt.scatter(df_test.ENGINESIZE, df_test.CO2EMISSIONS,  color='blue')
plt.plot(x_test[:,0], y_hat, '+r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Linear regression model using engine size, cylinders, fuel consumption in city and highway")
plt.legend(['prediction','actual'],loc='lower right')
plt.show()

# comparing the MSEs:
print("Addition of the fuel consumption parameters resulted in %.2f" %(abs(MSE_2-MSE_1)/MSE_1*100), "% improvement in mean squared error.")
