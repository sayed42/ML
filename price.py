from joblib import dump,load
import numpy as np
model = load('housing.joblib')

#the values of features are number of houses , tax , area crime rate , etc

features = np.array([[-0.42557196 , 3.12628155 , -1.12165014, -0.27288841, -1.42241324, -0.21966218, -1.31398765 , 2.60551661, -1.0016859 , -0.5778192 , -0.97491834 , 0.41594521, -0.84884195]])

print('The predicted price from the given features is : ',model.predict(features))

#As the values changes the predicted price will also change 
# 
# 
# change the values from numpy array in the features variable to see the new predicted price