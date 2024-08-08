

import pandas as pd
import pickle
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('random_forest.pkl','rb'))

st.title('Early Diabetes Prediction Using MachineLearning')
st.sidebar.subheader('User Input Parameters')

def user_input_paramters():
    Age= st.sidebar.number_input('Enter the Age')
    Gender = st.sidebar.selectbox('Gender:0-Female,1-Male',[0,1])
    Polyuria = st.sidebar.selectbox('Polyuria:0-No,1-Yes',[0,1])
    Polydipsia = st.sidebar.selectbox('Polydipsia:0-No,1-yes',[0,1])
    suddenweightloss = st.sidebar.selectbox('SuddenWeightLoss?:0-No,1-Yes',[0,1])
    weakness = st.sidebar.selectbox('weakness:0-No,1-Yes',[0,1])
    Polyphagia = st.sidebar.selectbox('Polyphagia:0-No,1-Yes',[0,1])
    Genitalthrush = st.sidebar.selectbox('Genitalthrush:,0-No,1-Yes',[0,1])
    visualblurring = st.sidebar.selectbox('visualblurring:,0-No,1-Yes',[0,1])
    itching = st.sidebar.selectbox('itching:0-No,1-Yes',[0,1])
    irritability = st.sidebar.selectbox('irritability:0-No,1-Yes',[0,1])
    delayedhealing = st.sidebar.selectbox('deleyedhealing:0-No,1-Yes',[0,1])
    PartialParesis = st.sidebar.selectbox('Partialparesis:0-No,1-Yes',[0,1])
    musclestiffness = st.sidebar.selectbox('musclestiffnes:0-No,1-Yes',[0,1])
    Alopecia = st.sidebar.selectbox('Alopecia:0-No,1-Yes',[0,1])
    Obesity = st.sidebar.selectbox('Obesity:0-No,1-Yes',[0,1])
    data = {'Age': Age,
           'Gender': Gender, 
           'Polyuria': Polyuria,
           'Polydipsia': Polydipsia,
           'sudden weight loss': suddenweightloss,
           'weakness': weakness,
           'Polyphagia': Polyphagia,
           'Genital thrush': Genitalthrush,
           'visual blurring': visualblurring,
           'Itching':itching,
           'Irritability':irritability,
           'delayed healing':delayedhealing,
           'partial paresis': PartialParesis,
           'muscle stiffness':musclestiffness,
           'Alopecia':Alopecia,
           'Obesity': Obesity}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_paramters()
st.subheader('User Input Paramters')
st.write(df)

pred = model.predict(df)
pred_prob = model.predict_proba(df)

st.subheader('Predicted Value:')
st.write(pred)






