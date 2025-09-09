import streamlit as st
import pandas as pd
import numpy as np

# --- App Title and Description ---
st.title("My First Streamlit App")
st.write("This is a simple app, to demonstrate the basic functionalities of Streamlit.")

# --- Interactive Widgets in the Sidebar ---
st.sidebar.header("User Input Features")

# Text Input
user_name = st.sidebar.text_input("What is your name?", "Jay reddy")

# Slider
age = st.sidebar.slider("Select your age", 0, 100, 25)

# Selectbox
favorite_color = st.sidebar.selectbox("What is your favorite color?", ["Blue", "Red", "Green", "Yellow"])

# --- Main Page Content ---
st.header(f"Welcome, {user_name}!")