# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:46:00 2018

@author: samka
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#loading data
data= pd.read_csv('carscom_scrape_clean.csv')
data = data.drop(['Unnamed: 0'],axis= 1)

#Dropping missing data primarly in mileage (noticed from data cleaning in R)
data= data.dropna()
data.columns

#Creating matrix of features and dependent variable
X= data.loc[:,['year', 'make', 'model', 'mileage', 'ratings']]
y= data['price']

#exploring correlation
corr= X.corr()
sns.heatmap(corr,
            xticklabels= corr.columns,
            yticklabels=corr.columns)

#Creating dummy variables
X= pd.get_dummies(data= X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Scaling data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.fit_transform(y_test)

#Gradient Boosting Regressor for predicting car prices
gbr = GradientBoostingRegressor(loss ='ls', max_depth=5, n_estimators = 300, learning_rate= 0.2 )

# fit the training data
gbr.fit (X_train, y_train)
# get the predicted values from the test set
predicted = gbr.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = gbr, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'loss': ['ls'], 'learning_rate': [0.2],
              'n_estimators':[200,300], 'max_depth':[5,6], 'verbose':[1] }]
            
grid_search = GridSearchCV(estimator = gbr,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10,
                          )
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Calculating R^2
r2 = r2_score(y_test,predicted)
print(r2)

#saving model
from sklearn.externals import joblib
joblib.dump(gbr, 'model.carscom')

#Predicting for a new instance
user_input = {'Year':2010, 'Mileage':80000, 'Ratings':5, 'Make':'Toyota', 'Model':'Tacoma'}

def input_to_one_hot(cars):
   # initialize the target vector with zero values
   enc_input = np.zeros(436)
   # set the numerical input as they are
   enc_input[0] = cars['Year']
   enc_input[1] = cars['Mileage']
   enc_input[2] = cars['Ratings']
   ##################### Mark #########################
   # get the array of make categories
   Make = data.make.unique()
   # redefine the the user inout to match the column name
   redefinded_user_input = 'make_' + cars['Make']
   # search for the index in columns name list
   Model_column_index = X.columns.tolist().index(redefinded_user_input)
   #print(mark_column_index)
   # fullfill the found index with 1
   enc_input[Model_column_index] = 1
   ##################### Fuel Type ####################
   # get the array of fuel type
   Model = data.model.unique()
   # redefine the the user inout to match the column name
   redefinded_user_input = 'model_'+ cars['Model']
   # search for the index in columns name list
   Model_column_index = X.columns.tolist().index(redefinded_user_input)
   # fullfill the found index with 1
   enc_input[Model_column_index] = 1
   return enc_input


a = input_to_one_hot(user_input)
price_pred = gbr.predict([a])
price_pred[0]



#Making a flask Api for deployment
from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

# load the built-in model 
gbr = joblib.load('model.carscom')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def get_delay():
    result=request.form
    year_model = result['year']
    mileage = result['mileage']
    make = result['make']
    ratings = result['ratings']
    # we create a json object that will hold data from user inputs
    user_input = {'year':year_model, 'mileage':mileage,  'make':make, 'ratings': ratings}
    # encode the json object to one hot encoding so that it could fit our model
    a = input_to_one_hot(user_input)
    # get the price prediction
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)
    # return a json value
    return json.dumps({'price':price_pred});

if __name__ == '__main__':
    app.run(port=8080, debug=True, use_reloader=False)
  

