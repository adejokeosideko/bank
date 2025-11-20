import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# LOAD DATA

data = pd.read_csv(r"C:\Users\DELL 7240\OneDrive\Documents\Financial gmc\Financial_inclusion_dataset (1).csv")

# ENCODING

data["country_Encoded"] = data["country"].map({"Rwanda":0, "Tanzania":1, "Kenya":2, "Uganda": 3})
data["bank_account_encoded"] = data["bank_account"].map({"No":0, "Yes":1})
data["location_type_encoded"] = data["location_type"].map({"Rural":0, "Urban":1})
data["cellphone_access_Encoded"] = data["cellphone_access"].map({"Yes":0, "No":1})
data["relationship_with_head_Encoded"] = data["relationship_with_head"].map({
    'Spouse':0,'Head of Household':1,'Other relative':2,'Child':3,'Parent':4,'Other non-relatives':5})
data["marital_status_encoded"] = data["marital_status"].map({
    'Married/Living together':0,'Widowed':1,'Single/Never Married':2,'Divorced/Seperated':3,'Dont know':4})
data["gender_of_respondent_Encoded"] = data["gender_of_respondent"].map({"Male":0,"Female":1})
data["education_level_Encoded"] = data["education_level"].map({
    "Primary education":0, "No formal education":1,"Secondary education":2,
    "Tertiary education":3,"Vocational/Specialised training":4,"Other/Dont know/RTA":5})
data["job_type_Encoded"]= data["job_type"].map({
    'Self employed':0,'Government Dependent':1,'Formally employed Private':2,'Informally employed':3,
    'Informally Self-employed':4,'Formally employed Government':5,'Farming and Fishing':6,
    'Remittance Dependent':7,'Other Income':8,'Dont Know/Refuse to answer':9,'No Income':10})

# FEATURES AND TARGET
X = data[['country_Encoded','location_type_encoded','cellphone_access_Encoded',
          'relationship_with_head_Encoded','marital_status_encoded','gender_of_respondent_Encoded',
          'education_level_Encoded','job_type_Encoded']]
y = data["bank_account_encoded"]


# TRAIN MODEL

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

LR_model = LogisticRegression(max_iter=2000)
LR_model.fit(X_scaled, y)

# STREAMLIT UI

st.title("Bank Account Prediction App")
st.subheader("Predict individuals most likely to have or use a bank account")
st.text("Dataset includes demographic and financial service information across East Africa.")

st.write("Fill the form below and click **Validate** to get a prediction.")

# Options dictionaries
country_Encoded = {"Rwanda":0, "Tanzania":1, "Kenya":2, "Uganda": 3}
location_type_encoded = {"Rural":0, "Urban":1}
cellphone_access_Encoded = {"Yes":0, "No":1}
relationship_with_head_Encoded = {'Spouse':0,'Head of Household':1,'Other relative':2,
                                  'Child':3,'Parent':4,'Other non-relatives':5}
marital_status_encoded = {'Married/Living together':0,'Widowed':1,'Single/Never Married':2,
                          'Divorced/Seperated':3,'Dont know':4}
gender_of_respondent_Encoded = {"Male":0,"Female":1}
education_level_Encoded = {"Primary education":0, "No formal education":1,"Secondary education":2,
                           "Tertiary education":3,"Vocational/Specialised training":4,
                           "Other/Dont know/RTA":5}
job_type_Encoded = {'Self employed':0,'Government Dependent':1,'Formally employed Private':2,
                    'Informally employed':3,'Informally Self-employed':4,
                    'Formally employed Government':5,'Farming and Fishing':6,
                    'Remittance Dependent':7,'Other Income':8,
                    'Dont Know/Refuse to answer':9,'No Income':10}

# INPUTS
country = st.selectbox("Country", list(country_Encoded.keys()))
location = st.selectbox("Location Type", list(location_type_encoded.keys()))
cellphone_access = st.selectbox("Cellphone Access", list(cellphone_access_Encoded.keys()))
relationship_with_head = st.selectbox("Relationship to Head of Household", list(relationship_with_head_Encoded.keys()))
marital_status = st.selectbox("Marital Status", list(marital_status_encoded.keys()))
gender = st.selectbox("Gender", list(gender_of_respondent_Encoded.keys()))
education_level = st.selectbox("Education Level", list(education_level_Encoded.keys()))
job_type = st.selectbox("Job Type", list(job_type_Encoded.keys()))

# CREATE INPUT DF
input_df = pd.DataFrame({
    "country_Encoded":[country_Encoded[country]],
    "location_type_encoded":[location_type_encoded[location]],
    "cellphone_access_Encoded":[cellphone_access_Encoded[cellphone_access]],
    "relationship_with_head_Encoded":[relationship_with_head_Encoded[relationship_with_head]],
    "marital_status_encoded":[marital_status_encoded[marital_status]],
    "gender_of_respondent_Encoded":[gender_of_respondent_Encoded[gender]],
    "education_level_Encoded":[education_level_Encoded[education_level]],
    "job_type_Encoded":[job_type_Encoded[job_type]]
})

# VALIDATE BUTTON
if st.button("Validate"):
    input_scaled = scalar.transform(input_df)
    prediction = LR_model.predict(input_scaled)

    if prediction == 1:
        st.success(" This person is likely to have or use a bank account!")
        st.balloons()
    else:
        st.error("This person is NOT likely to have or use a bank account.")
