import streamlit as st
from streamlit_chat import message


def main():
    st.set_page_config(page_title="Pharmogle", page_icon="💊")

    st.header("Pharmogle 💊")
    st.divider()

    message("👋 Hello! Welcome to Pharmogle! How can I help you today with your medicine-related questions?")
    message("What is the dosage of Panadol?", is_user=True)

    st.text_input("Ask a question about your medicine:")

    with st.sidebar:
        st.subheader("Your Medicine Picture")
        uploaded_file = st.file_uploader("Upload your Picture here and click on 'Process'")
        st.button("Process", type="primary")

if __name__ == '__main__':
    main()