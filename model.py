import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings('ignore')


house = pd.read_csv('house.csv')
house.head(5)

var = ['TAX', 'FLOORS', 'ROOMS']

house_X = house[var]             # Define the X to be used (i.e., based on selected var)
house_y = house['TOTAL_VALUE']   # Define the Y



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(house_X, house_y)

# Saving model to disk using pickle 

pickle.dump(regressor, open('model.pkl', 'wb' ))

# Loading model to compare the results
                        
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4330,2,6]]))
                           