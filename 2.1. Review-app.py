#"""
#This code builds a streamlit app which takes review as input and spits out the underlying conditon and top 5 drug Recommendation for the condition according with single submit.
#"""

import streamlit as st
import pickle
import pandas as pd

#Load the dataframe and model from a pickle file
with open('df_app.pkl', 'rb') as f:
    df = pd.read_pickle(f)

with open('model.pkl', 'rb') as f:
    pass_tf2, tfidf_vectorizer2 = pickle.load(f)

# Set the title
st.title("Matching Patients with Effective Medications")

# Set Header Title
st.header('**Welcome to our health prediction app!** :wave:')

# Set the description
st.markdown("""
Get quick and accurate results for your health condition with our app. Type in your review, click Submit, and let our advanced algorithms do the rest. We'll identify your underlying condition and provide you with the top 5 drugs for relief. Try it today and feel better :smile:.
""")

#create a text input for user input
#text_input = st.text_input('Type your review here')
text_input = st.text_area('Type your review here', height=100)

#create a submit button
submit_button = st.button('Submit')

# Input for the function
prediction = None
if submit_button and text_input:
    prediction = pass_tf2.predict(tfidf_vectorizer2.transform([text_input]))[0]
    st.write('Based on the review given this could be your underlying condition:', prediction)

option = prediction

#Define a function to display a different part of the dataframe based on the option selected
def display_dataframe(option):
    if option == 'Depression':
        df_filtered = df[df['condition'] == 'Depression'].tail(5).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)
    elif option == 'Diabetes, Type 2':
        df_filtered = df[df['condition'] == 'Diabetes, Type 2'].tail(5).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)
    else:
        df_filtered = df[df['condition'] == 'High Blood Pressure'].tail(5).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)

if prediction:
    display_dataframe(prediction)

