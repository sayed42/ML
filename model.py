import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer

housing = pd.read_csv("data.csv")



#train_set , test_set = train_test_split(housing,test_size = 0.2, random_state = 42)
#print(f"rows in train set: {len(train_set)}\n rows in test set: {len(test_set)}")


from sklearn.model_selection import StratifiedShuffleSplit
cvs = StratifiedShuffleSplit(n_splits=1 , test_size= 0.2, random_state=42)
for train_index, test_index in cvs.split(housing,housing['CHAS']):
    strat_train = housing.loc[train_index]
    strat_test = housing.loc[test_index]
#print(strat_test['CHAS'].value_counts())

#removing price column from the data because we are considering it as a label

housing = strat_train.drop('price',axis=1)
housing_label = strat_train['price'].copy()


#To add median value in empty(NaN) values so that the model runs properly if value are missing from the data



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scalar',StandardScaler()),
])

housing_num = my_pipeline.fit_transform(housing)
#print(housing_num)





#to check which model is best suited for data .. I'm using RandomForestregressor but you can also use LinearRegressor or DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num,housing_label)

some_data = housing.iloc[:5]
some_label = housing_label.iloc[:5]

prepared_data= my_pipeline.transform(some_data)

model.predict(prepared_data)
a = list(some_label)
#print(a)

#root mean squared error is used to find the error in the predictions . Smaller the value(eg : 2.1 , 1.5 , 2.3 ,etc) better the model will be

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_label,housing_predictions)
rmse = np.sqrt(mse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num,housing_label,scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)
#print(rmse_scores)

def print_scores(scores):
    print('Scores : ',scores)
    print('Mean : ',scores.mean())
    print('Standard  deviation : ',scores.std())
#print_scores(rmse_scores)


##to save the program as a joblib file format to so that we can run it on another .py file
#from joblib import dump, load
#dump(model,'housing.joblib')


#to test the predicted vs real values in the data
x_test = strat_test.drop('price',axis=1)
y_test = strat_test['price'].copy()
xtest_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(xtest_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print('Real values : \n',list(y_test))
print('\nPredicted values : \n',final_predictions)
