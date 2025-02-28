import streamlit as st
st.title("Hello, Streamlit! 🎈")
st.write("This is a simple web app built with Streamlit.")
name = st.text_input("Enter your name:")
if st.button("Greet"):
    st.write(f"Hello, {name}! 👋")
