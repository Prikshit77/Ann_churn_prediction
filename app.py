import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle

## loading the trained model

model=tf.keras.models.load_model('churn_model.keras')

## load the endor and scalar
with open('labelencoder_gender.pkl','rb') as file:
    labelencoder=pickle.load(file)

with open('ohe_geo.pkl','rb') as file:
    ohe_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# steamlit app
st.title("Customer Churn Prediction")

# user input
CreditScore=st.number_input('Credit Score')
Geography=st.selectbox('Geography',ohe_geo.categories_[0])
Gender=st.selectbox('Gender',labelencoder.classes_)
Age=st.slider('Age',18,95)
Tenure=st.slider('Tenure',0,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('No. of Products',1,4)
HasCrCard=st.selectbox('Has Credit Card',[0,1])
IsActiveMember=st.selectbox('Is Active Member',[0,1])
EstimatedSalary=st.number_input('Estimated Salary')

## Peparing the input data
input_data = pd.DataFrame({
    'CreditScore':[CreditScore] ,
    'Gender': [labelencoder.transform([Gender])[0]],
    'Age': [Age],
    'Tenure':[Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
}
)

## one hot encoding 'Geography'
geo_encoded=ohe_geo.transform([[Geography]]).toarray()
geo_encoded=pd.DataFrame(geo_encoded,columns=ohe_geo.get_feature_names_out(['Geography']))

## concatinating the data
input_data=pd.concat([input_data,geo_encoded],axis=1)

## standardization with scalar
input_data_scaled=scaler.transform(input_data)

## predict churn
prediction=model.predict(input_data_scaled)
prediction

prediction_prob = prediction[0][0]

if prediction_prob>0.5:
    st.write("customer will likely churn")
else:
    st.write("customer will not churn")