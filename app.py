import streamlit as st
from streamlit_chat import message as st_message
import pandas as pd
import numpy as np
import datetime
import pickle
import os
import csv
import json
import torch
from tqdm.auto import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Clarifai
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import utils
import toml

# secrets = toml.load(".streamlit\secrets.toml")["secrets"]

PAT = st.secrets["PAT"]
Llama_USER_ID = st.secrets["Llama_USER_ID"]
Llama_APP_ID = st.secrets["Llama_APP_ID"]
Llama_MODEL_ID = st.secrets["Llama_MODEL_ID"]

USER_ID = st.secrets["MC_USER_ID"]
APP_ID = st.secrets["MC_APP_ID"]
MODEL_ID = st.secrets["MC_MODEL_ID"]
MODEL_VERSION_ID = st.secrets["MC_MODEL_VERSION_ID"]


prompt_template = """

You are an expert Pharmacist Bot. Your job is to answer user queries and help them understand their medication. While answering the user, use the following context provided to you. If you can't answer the user query using the context, please respond with "I don't know the answer to this, please consult you physician". You must never generate false information.
Always make sure to provide short, clear, consise and useful information. If listing down important points, use a bullet point list. Never answer with any unfinished response. Do not refer to yourself in the responses.

{context}

Question: {question}

Always make sure to provide short, clear, consise and useful information. If listing down important points, use a bullet point list. Never answer with any unfinished response. Do not refer to yourself in the responses.

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


st.set_page_config(
    page_title = 'PharmOgle Bot 🤖',
    page_icon = '🤖')


@st.cache_resource
def load_embedding_model():
    embedding_model = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base',
                                                model_kwargs = {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')})
    return embedding_model

@st.cache_data
def load_faiss_index(medicine="adol"):
    vector_database = FAISS.load_local("indexes/"+medicine, embedding_model)
    return vector_database

@st.cache_resource
def load_llm_model():
    llm = Clarifai(
    pat=PAT, user_id=Llama_USER_ID, app_id=Llama_APP_ID, model_id=Llama_MODEL_ID, verbose=True
    )

    return llm


def load_retriever(llm, db):
    qa_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                            retriever=db.as_retriever(),
                            chain_type_kwargs= chain_type_kwargs)

    return qa_retriever

def retrieve_document(query_input):
    related_doc = vector_database.similarity_search(query_input)
    return related_doc

def retrieve_answer():
    prompt_answer=  st.session_state.my_text_input + " " + "Please provide clear, concise and useful information."
    answer = qa_retriever.run(prompt_answer)
    log = {"timestamp": datetime.datetime.now(),
        "question":st.session_state.my_text_input,
        "generated_answer": answer[6:],
        "rating":0 }

    st.session_state.history.append(log)
    st.session_state.chat_history.append({"message": st.session_state.my_text_input, "is_user": True})
    if len(answer) > 0:
        st.session_state.chat_history.append({"message": answer, "is_user": False, "avatar_style": "bottts-neutral", "seed": "Princess"})
    else:
        st.session_state.chat_history.append({"message": "I'm sorry there was a technical glitch and I couldn't fetch an answer for you :('", "is_user": False, "avatar_style": "bottts-neutral", "seed": "Princess"})

    st.session_state.my_text_input = ""

    return answer[6:]


def clean_chat_history():
    st.session_state.chat_history = []


if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
datetime_format= "%Y-%m-%d %H:%M:%S"

embedding_model = load_embedding_model()
llm_model = load_llm_model()
    

st.write("# PharmOgle-Bot 🤖")
st.markdown("""
         The PharmOgle Bot project is a virtual pharmacist developed by team **enGenAIr** for the **Lablab.ai Llama2+Clarifai hackathon**. The bot provides quick access to information about different pharmacuetical products. The bot is powered by the Llama2 language model and the Clarifai image recognition model.
         The goal of the PharmOgle Bot is to provide an easy and user-friendly way for everyone to access information about the medication they're taking, possible side-effects or any other questions they might have.  
          """)
st.write(" ")
st.write("""For the purpose of this demo, the bot has been trained to classify () drugs. You can either choose the drug name from the dropdown menu or upload an image of the drug packaging. The bot will then answer any questions you have about the drug.

""")
st.write(' ⚠️ Please expect to wait **~ 10 - 20 seconds per question**.')
st.write(' ⚠️ **Disclaimer:** This should not be considered as an alternate to professional medical advice.')

st.markdown("---")
st.write(" ")

uploaded_file = None
demo_option = st.radio(
    "Choose an option",
    ["Upload my own image","Choose from list"]
)
if demo_option == 'Choose from list':
    image_label = st.selectbox(
    'Choose your medication',
    ('Adol','Aggrex','Amrizole','Atoreza','Augmentin','Betadine','Brufen','C-retard','Ceftriaxone','Celebrex','Cemicresto','Cholerose','Ciprofar','Clarinase','Congestal','Daflon','Dalacin','Diflucan','Flagyl','Floxabact','Foradil','Fucidin','Garamycin','Glucophage','Ivypront','Janumet','Jusprin','Lactulose','Lamifen','Megamox','Midodrine','Mucophylline','Neurovit','Oracure','Pridocaine','Primrose','Sediproct','Zantac','Zyrtec'), key = "mediselect")
    

    vector_database = load_faiss_index(image_label.lower())
    qa_retriever = load_retriever(llm= llm_model, db= vector_database)

    st.write("""
            ### Ask a question
            """)


    for chat in st.session_state.chat_history:
        st_message(**chat)

    query_input = st.text_input(label= 'Please Enter your Question about '+image_label , key = 'my_text_input', on_change= retrieve_answer)

    clear_button = st.button("Start new convo",
                            on_click=clean_chat_history,
                            key="mediclear")

    st.write(" ")
    st.write(" ")
else:
    # get image code
    uploaded_file = st.file_uploader(label='**Upload Image file**', type=['jpg','jpeg','png'], accept_multiple_files=False)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_label = utils.get_image_label(bytes_data, USER_ID, APP_ID, MODEL_ID, PAT, MODEL_VERSION_ID)
        vector_database = load_faiss_index(image_label)
        qa_retriever = load_retriever(llm= llm_model, db= vector_database)

        st.write("""
                ### Ask a question
                """)


        for chat in st.session_state.chat_history:
            st_message(**chat)

        query_input = st.text_input(label= 'Please Enter your Question about '+image_label.capitalize() , key = 'my_text_input', on_change= retrieve_answer )

        clear_button = st.button("Start new convo",
                                on_click=clean_chat_history)

        st.write(" ")
        st.write(" ")

st.markdown("---")

