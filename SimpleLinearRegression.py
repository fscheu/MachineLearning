# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# # Simple Linear Regression
# 
# Estimated time needed: **15** minutes
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# -   Use scikit-learn to implement simple Linear Regression
# -   Create a model, train,test and use the model
# 
# %% [markdown]
# ### Importing Needed packages
# 

# %%
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ### Downloading Data
# 
# To download the data, we will use !wget to download it from IBM Object Storage.
# 

# %%
get_ipython().system('wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv')

# %% [markdown]
# **Did you know?** When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)
# 
# %% [markdown]
# ## Understanding the Data
# 
# ### `FuelConsumption.csv`:
# 
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
# 
# -   **MODELYEAR** e.g. 2014
# -   **MAKE** e.g. Acura
# -   **MODEL** e.g. ILX
# -   **VEHICLE CLASS** e.g. SUV
# -   **ENGINE SIZE** e.g. 4.7
# -   **CYLINDERS** e.g 6
# -   **TRANSMISSION** e.g. A6
# -   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# -   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# -   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# -   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 
# %% [markdown]
# ## Reading the data in
# 

# %%
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

# %% [markdown]
# ### Data Exploration
# 
# Lets first have a descriptive exploration on our data.
# 

# %%
# summarize the data
df.describe()

# %% [markdown]
# Lets select some features to explore more.
# 

# %%
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# %% [markdown]
# We can plot each of these fearues:
# 

# %%
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# %% [markdown]
# Now, lets plot each of these features vs the Emission, to see how linear is their relation:
# 

# %%
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# %%
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# %% [markdown]
# ## Practice
# 
# Plot **CYLINDER** vs the Emission, to see how linear is their relation:
# 

# %%
# write your code here


# %% [markdown]
# <details><summary>Click here for the solution</summary>
# 
# ```python
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()
# 
# ```
# 
# </details>
# 
# %% [markdown]
# #### Creating train and test dataset
# 
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.
# 
# This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.
# 
# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We create a mask to select random rows using **np.random.rand()** function: 
# 

# %%
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# %% [markdown]
# ### Simple Regression Model
# 
# Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the actual value y in the dataset, and the predicted value yhat using linear approximation. 
# 
# %% [markdown]
# #### Train data distribution
# 

# %%
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# %% [markdown]
# #### Modeling
# 
# Using sklearn package to model data.
# 

# %%
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# %% [markdown]
# As mentioned before, **Coefficient** and **Intercept** in the simple linear regression, are the parameters of the fit line. 
# Given that it is a simple linear regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from our data. 
# Notice that all of the data must be available to traverse and calculate the parameters.
# 
# %% [markdown]
# #### Plot outputs
# 
# %% [markdown]
# We can plot the fit line over the data:
# 

# %%
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% [markdown]
# #### Evaluation
# 
# We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
# 
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: 
# 
# ```
# - Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# - Root Mean Squared Error (RMSE).
# - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# ```
# 

# %%
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

# %% [markdown]
# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="https://www.ibm.com/analytics/spss-statistics-software">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://www.ibm.com/cloud/watson-studio">Watson Studio</a>
# 
# %% [markdown]
# ### Thank you for completing this lab!
# 
# ## Author
# 
# Saeed Aghabozorgi
# 
# ### Other Contributors
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                 |
# | ----------------- | ------- | ------------- | ---------------------------------- |
# | 2020-11-03        | 2.1     | Lakshmi Holla | Changed URL of the csv             |
# | 2020-08-27        | 2.0     | Lavanya       | Moved lab to course repo in GitLab |
# |                   |         |               |                                    |
# |                   |         |               |                                    |
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

