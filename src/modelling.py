import pandas as pd
import numpy as np

# encoders and models
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from category_encoders import BinaryEncoder

# error metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
from statsmodels.tools.eval_measures import rmse

# visualizations
import plotly.express as ex
import plotly.offline as po
from prophet.plot import plot_plotly,plot_components_plotly
import matplotlib.pyplot as plt

# webapp
import streamlit as st

# others
import os
import warnings
warnings.filterwarnings('ignore')

##model loading
import joblib 

# loading data
path_train = r"C:\Users\DAVID\Career_Accelerator_LP4_ML-Appl\dataset_streamlit\train.csv"
train= pd.read_csv(path_train,usecols=['ds','holiday','locale','transferred','y','onpromotion','transactions'])
train.head()
train_target=  train[['y']]

path_test = r"C:\Users\DAVID\Career_Accelerator_LP4_ML-Appl\dataset_streamlit\test.csv"
test= pd.read_csv(path_test,usecols=['ds','holiday','locale','transferred','onpromotion','transactions'])
test.head()

path_test_with_y = r"C:\Users\DAVID\Career_Accelerator_LP4_ML-Appl\dataset_streamlit\test_with_y.csv"
test_with_y= pd.read_csv(path_test_with_y)
test_with_y.head()
test_target=test_with_y[['sales']]

# encoding
train.info()
train['transferred']=train['transferred'].astype(object)
# making a copy
train_copy= train.copy()
#categorical columns
##train_cat=train_copy.select_dtypes(np.object)
#train_cat.columns

# encoding the holiday column
BE=BinaryEncoder(cols='holiday')
train_copy= BE.fit_transform(train_copy)
# encoding the locale
rank= ["National", "Regional", "Local", "Not Holiday"]
# instantiating the encoder
OE=OrdinalEncoder(categories=[rank])
train_copy[['locale']]=OE.fit_transform(train_copy[['locale']])
# encoding the transferred column
# instantiating
LE= LabelEncoder()
train_copy['transferred']=LE.fit_transform(train_copy['transferred'])
# dropping the column y
train_copy=train_copy.drop('y',axis=1)
train_copy.head()

# modelling train
model=Prophet(yearly_seasonality=True, seasonality_mode='multiplicative',seasonality_prior_scale=20)
# stating exogenous variables
exo_cols= [ 'holiday_0', 'holiday_1', 'holiday_2', 'locale', 'transferred', 'onpromotion', 'transactions']

model.add_regressor(exo_cols[0], standardize=True)
model.add_regressor(exo_cols[1], standardize=True)
model.add_regressor(exo_cols[2], standardize=True)
model.add_regressor(exo_cols[3], standardize=True)
model.add_regressor(exo_cols[4], standardize=True)
model.add_regressor(exo_cols[-1], standardize=True)
# concatenating 
full_train= pd.concat([train_copy,train_target],axis=1)
full_train.head()

# fitting the model
model.fit(full_train)

#making a copy
test_copy= test.copy()
test_copy.head()

# encoding the holiday column
BE=BinaryEncoder(cols='holiday')
test_copy= BE.fit_transform(test_copy)

# encoding the locale
rank= ["National", "Regional", "Local", "Not Holiday"]
# instantiating the encoder
OE=OrdinalEncoder(categories=[rank])
test_copy[['locale']]=OE.fit_transform(test_copy[['locale']])

# encoding the transferred column
# instantiating
LE= LabelEncoder()
test_copy['transferred']=LE.fit_transform(test_copy['transferred'])
test_copy.head()

#Prediction
eval_fp= model.predict(test_copy)
eval_fp.head()
eval=eval_fp[['yhat']]
eval.head()

# Model_Performance
mae=(mean_absolute_error(eval,test_target)/test_target.mean())*100
rsmle=np.sqrt(mean_squared_log_error(eval,test_target))
rmse=np.sqrt(mean_squared_error(eval,test_target))

result= pd.DataFrame({'MAE':mae,'RSMLE':rsmle,'RSME':rmse})
result

# visualizing the outcome
model.plot(eval_fp)
plt.show()

# taking sample
# making copy
train_2= train.copy()
train_2=train_2.drop(['holiday','transferred','locale'],axis=1)
# making copy
test_2= test.copy()
test_2=test_2.drop(['holiday','transferred','locale'],axis=1)

#modelling
# instantiating the model
model_2= Prophet(yearly_seasonality=True, seasonality_mode='multiplicative',seasonality_prior_scale=20)
# adding holiday feature
model_2.add_country_holidays(country_name='Ecuador')
# adding regressors (exogenous variable)
cols_to_add = [col for col in train_2.drop(['ds', 'y'], axis=1)]
[model_2.add_regressor(col, standardize=True) for col in cols_to_add]
#for col in train_2.drop(['ds','y'],axis=1):
#    model_2.add_regressor(col,standardize=True)
model_2.fit(train_2)

eval_f=model_2.predict(test_2)
# predicted values
eval_2= eval_f[['yhat']]

# Performance
mae=(mean_absolute_error(eval_2,test_target)/test_target.mean())*100
rsmle=np.sqrt(mean_squared_log_error(eval_2,test_target))
rmse=np.sqrt(mean_squared_error(eval_2,test_target))

res= pd.DataFrame({'MAE':mae,'RSMLE':rsmle,'RSME':rmse})
res
# visualization of the components
model_2.plot_components(eval_f)

##saving my Facebook Prophet model
joblib.dump(model_2,r"C:\Users\DAVID\Career_Accelerator_LP4_ML-Appl\ml_component\saved_ml.joblib")












