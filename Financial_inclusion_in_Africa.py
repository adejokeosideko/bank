import pickle
import streamlit as st
import pandas as pd
with open("LR_model.pkl", "rb") as f:
    LR_model= pickle.load(f)
with open("scalar.pkl", "rb")as f:
    scalar = pickle.load(f)

#encoding dictionaries
country_Encoded = {"Rwanda":0, "Tanzania":1, "Kenya":2, "Uganda": 3}
location_type_encoded = {"Rural":0, "Urban":1}
cellphone_access_Encoded = {"Yes":0, "No":1}
relationship_with_head_Encoded ={'Spouse':0,'Head of Household':1,
                            'Other relative':2,'Child':3,'Parent':4,
                             'Other non-relatives':5}
marital_status_encoded = {'Married/Living together':0,'Widowed':1,
                        'Single/Never Married':2,'Divorced/Seperated':3,
                          'Dont know':4}
gender_of_respondent_Encoded = {"Male":0,"Female":1}
education_level_Encoded = {"Primary education":0, "No formal education":1,
                          "Secondary education": 2,"Tertiary education":3,
                         "Vocational/Specialised training": 4,
                           "Other/Dont know/RTA": 5}
job_type_Encoded = {'Self employed':0,'Government Dependent':1,'Formally employed Private':2,
                  'Informally employed':3,'Informally Self-employed':4,
                  'Formally employed Government':5,'Farming and Fishing':6,
                  'Remittance Dependent':7,'Other Income':8,
                  'Dont Know/Refuse to answer':9,'No Income':10}
st.title("Bank account prediction")
st.image("Screenshot 2025-11-15 165813.png", width = 150)
st.header("This is a machine learning algorithm")
st.subheader("It predicts the individuals who are" \
" most likely to have or use a bank account")
st.text("The dataset used for this algorithm"\
        "contains demographic information and the  financial services that" \
        " are used by approximately 33,600 individuals across East Africa.")

st.write("Fill the form below and click Validate to get prediction")

country = st.selectbox("Country",
                       options = list(country_Encoded.keys()))
location = st.selectbox("Choose your type of Location", 
                        options= list(location_type_encoded.keys()))
cellphone_access = st.selectbox(" Do you have access to cellphone",
                               options = list(cellphone_access_Encoded.keys()))
relationship_with_head = st.selectbox("Relationship with the Head of the Household",
                                      options = list(relationship_with_head_Encoded.keys()))
marital_status = st.selectbox("Marital status",
                              options = list(marital_status_encoded.keys()))
gender_of_respondent = st.selectbox("Gender", 
                                    options = list(gender_of_respondent_Encoded.keys()))
education_level = st.selectbox("Highest level of Education",
                               options = list(education_level_Encoded.keys()))
job_type = st.selectbox("Type of job",
                       options= list(job_type_Encoded.keys()))

#putting all the inputs in a datafram
input_df = pd.DataFrame({"country_Encoded":[country_Encoded[country]],
                         "location_type_encoded": [location_type_encoded[location]],
                         "cellphone_access_Encoded":[cellphone_access_Encoded[cellphone_access]],
                         "relationship_with_head_Encoded":[relationship_with_head_Encoded[relationship_with_head]],
                         "marital_status_encoded":[marital_status_encoded[marital_status]],
                         "gender_of_respondent_Encoded":[gender_of_respondent_Encoded[gender_of_respondent]],
                         "education_level_Encoded":[education_level_Encoded[education_level]],
                         "job_type_Encoded":[job_type_Encoded[job_type]]})
#predict button
if st.button("Validate"):
    #scale the data
    input_scaled = scalar.transform(input_df)

    #Make  prediction
    pred = LR_model.predict(input_scaled)
    #display the result
    if pred ==1:
        st.success("Awesome!!! This person is likely to have or use a bank account")
        st.ballon()
    else:
        st.error("This person is NOT likely to have or use a bank account")