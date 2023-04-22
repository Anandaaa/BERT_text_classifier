import streamlit as st
import requests


st.title("BERT text classifier")

st.text("Hello user. Welcome to the text classification service!")
input_text = st.text_input('input text', 'hello, how is it going')

if input_text:
    req = requests.post("http://localhost:8000/score", json={'data': input_text})
    prob_string = '\n\t'.join([k+': '+str(v) for k, v in req.json().items()])
    st.text(f"text classification model predictions: \n\t{prob_string}")
