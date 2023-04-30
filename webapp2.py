import streamlit as st
import pandas as pd
import numpy as np

# Load the dataframe from a pickle file
with open('df_app.pkl', 'rb') as f:
    df = pd.read_pickle(f)

# Define the dropdown menu options
options = ['Depression', 'Type 2 Diabetes', 'High Blood Pressure']

# Define a function to display a different part of the dataframe based on the option selected
def display_dataframe(option, tail):
    if option == 'Depression':
        df_filtered = df[df['condition'] == 'Depression'].tail(tail).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)
    elif option == 'Type 2 Diabetes':
        df_filtered = df[df['condition'] == 'Diabetes, Type 2'].tail(tail).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)
    else:
        df_filtered = df[df['condition'] == 'High Blood Pressure'].tail(tail).iloc[::-1].reset_index(drop=True)
        df_filtered.index += 1
        st.table(df_filtered)

# Create the Streamlit app
st.title('Best Drug Suggestion for Medical disorder ')
#st.write('## Final dataframe after sentiment analysis')
#st.write(df)
option = st.selectbox('Select a condition:', options)
tail = st.number_input('Enter the number of top drugs you want to know for the disorder:', min_value=1, max_value=len(df), value=5, step=1)
if st.button('Submit'):
    display_dataframe(option, tail)



