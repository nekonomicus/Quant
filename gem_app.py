# gem_app.py (Temporary Minimal Test)
import streamlit as st
import os

st.set_page_config(page_title="Render Test")
st.title("Hello from Render!")
st.write("If you see this, Streamlit and Render are configured correctly.")
st.balloons()
st.write(f"Running on port: {os.environ.get('PORT')}") # Check if Render's PORT is accessible
