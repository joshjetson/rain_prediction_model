#!/usr/bin/env python
# This dataset contains observations of weather metrics for each day from 2008 to 2017. The **weatherAUS.csv** dataset includes the following fields:
# 
# | Field         | Description                                           | Unit            | Type   |
# | ------------- | ----------------------------------------------------- | --------------- | ------ |
# | Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
# | Location      | Location of the Observation                           | Location        | object |
# | MinTemp       | Minimum temperature                                   | Celsius         | float  |
# | MaxTemp       | Maximum temperature                                   | Celsius         | float  |
# | Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
# | Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
# | Sunshine      | Amount of bright sunshine                             | hours           | float  |
# | WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
# | WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
# | WindDir9am    | Wind direction averaged of 10 minutes prior to 9am    | Compass Points  | object |
# | WindDir3pm    | Wind direction averaged of 10 minutes prior to 3pm    | Compass Points  | object |
# | WindSpeed9am  | Wind speed averaged of 10 minutes prior to 9am        | Kilometers/Hour | float  |
# | WindSpeed3pm  | Wind speed averaged of 10 minutes prior to 3pm        | Kilometers/Hour | float  |
# | Humidity9am   | Humidity at 9am                                       | Percent         | float  |
# | Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
# | Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
# | Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
# | Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
# | Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
# | Temp9am       | Temperature at 9am                                    | Celsius         | float  |
# | Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
# | RainToday     | If there was rain today                               | Yes/No          | object |
# | RISK_MM       | Amount of rain tomorrow                               | Millimeters     | float  |
# | RainTomorrow  | If there is rain tomorrow                             | Yes/No          | float  |
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


# ### Importing the Dataset
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv')

df.head()


# ### Data Preprocessing
# #### Transforming Categorical Variables
# First, we need to convert categorical variables to binary variables. We will use pandas `get_dummies()` method for this.
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])


# Next, we replace the values of the 'RainTomorrow' column changing them from a categorical column to a binary column. We do not use the `get_dummies` method because we would end up with two columns for 'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


# ### Training Data and Test Data
# Now, we set our 'features' or x values and our Y or target variable.
df_sydney_processed.drop('Date',axis=1,inplace=True)

df_sydney_processed = df_sydney_processed.astype(float)

features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']


# Linear Regression
# We use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `10`.
# 

x_train, x_test, y_train, y_test = train_test_split( features, Y, test_size=0.2, random_state=10)

LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)


# Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.

predictions = LinearReg.predict(x_test)


# Using the `predictions` and the `y_test` dataframe we calculate the value for each metric using the appropriate function.


LinearRegression_MAE = np.mean(np.absolute(predictions - y_test))
LinearRegression_MSE = np.mean((predictions - y_test)**2)
from sklearn.metrics import r2_score
LinearRegression_R2 = r2_score(y_test, predictions)


# Now we show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.


Report = {"Metrics":["MAE","MSE","R2"],"Result": 
    [LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2]}
pd.DataFrame(Report)


# KNN 

# Now we create and train a KNN model called KNN using the training data (`x_train`, `y_train`) with the `n_neighbors` parameter set to `4`.


k = 4
KNN = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)


# Now we use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.


predictions = KNN.predict(x_test)


# Now Using the `predictions` and the `y_test` dataframe we calculate the value for each metric using the appropriate function.


KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions,pos_label=0)
KNN_F1_Score = f1_score(y_test, predictions, average='weighted')
Report = {"Metrics":["KNN Accuracy","Jaccard","F1"],"Result": 
    [KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score]}
pd.DataFrame(Report)


# ### Decision Tree Portion
# We create and train a Decision Tree model called Tree using the training data (`x_train`, `y_train`).


Tree = DecisionTreeClassifier(criterion='entropy',max_depth=4)
Tree.fit(x_train, y_train)
# Now we use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`.
predictions = Tree.predict(x_test) 


Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions,pos_label=0)
Tree_F1_Score = f1_score(y_test, predictions, average='weighted')
Report = {"Metrics":["Tree Accuracy","Jaccard","F1"],"Result": 
    [Tree_Accuracy_Score,Tree_JaccardIndex,Tree_F1_Score]}
pd.DataFrame(Report)


# ### Logistic Regression
x_train, x_test, y_train, y_test = train_test_split( features, Y, test_size=0.2, random_state=1) 


LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train) 


predictions = LR.predict(x_test)

LR_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions,pos_label=0)
LR_F1_Score = f1_score(y_test, predictions, average='weighted')
LR_Log_Loss = log_loss(y_test, predictions)
Report = {"Metrics":["LR Accuracy","Jaccard","F1", "Log Loss"],"Result": 
    [LR_Accuracy_Score,LR_JaccardIndex,LR_F1_Score, LR_Log_Loss]}
pd.DataFrame(Report)


# SVM


SVM = svm.SVC(kernel='rbf').fit(x_train, y_train) 
predictions = SVM.predict(x_test) 
SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions) 
SVM_JaccardIndex = jaccard_score(y_test, predictions,pos_label=0) 
SVM_F1_Score = f1_score(y_test, predictions, average='weighted')


# Report


Report = {"Model":["KNN","Tree","LR","SVM"],
          "Accuracy":[KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
          "Jaccard": [KNN_JaccardIndex,Tree_JaccardIndex,LR_JaccardIndex,SVM_JaccardIndex ],
          "F1":[KNN_F1_Score,Tree_F1_Score,LR_F1_Score,SVM_F1_Score],
         "Log Loss":["N/A", "N/A", LR_Log_Loss, "N/A"]}
pd.DataFrame(Report)
