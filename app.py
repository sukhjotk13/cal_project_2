import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
#Not gonna use the visualization libraries for front end
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#load data
cal=fetch_california_housing()
df=pd.DataFrame(cal.data,columns=cal.feature_names)
df['Price']=cal.target

#title of the app
st.title("California Housing Price Prediction")

#Data Overview
st.header("Data overview for the first ten rows")
st.write(df.head(10))


#Ctrl+Z to stop the last instance- Whatever you write in st is gonna be shown on the front end
X = df.drop(columns=["Price"]) # input features
y = df['Price'] # target
X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

#Standardize the data 
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#model selection
mod=st.selectbox("Select a model",["Linear Regression","Random Forest","Decision Tree"])

models={
    "Linear Regression":LinearRegression(),
    "Random Forest":RandomForestRegressor(),
    "Decision Tree":DecisionTreeRegressor()
}

#Train the model
selected_model=models[mod] #Initializing the selected model

#Train the selected model
selected_model.fit(X_train,y_train)
y_pred=selected_model.predict(X_test)

#Model Evaluation
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

#Display the results
st.write(f"R2 Score: {r2}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")

st.write("Enter the input values for prediction:")

user_input={}
for column in X.columns:
    user_input[column]=[st.number_input(column,min_value=np.min(X[column]),max_value=np.max(X[column]),value=np.mean(X[column]))]

#convert dictionary into df
user_input_df=pd.DataFrame(user_input)

#Standardize the user input
user_input_sc_df=scaler.transform(user_input_df)

predicted_price=selected_model.predict(user_input_df)

st.write(f"Predicted price of the house:{predicted_price[0]*100000}")
