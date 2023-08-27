from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Clarifai
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR
import toml

secrets = toml.load(".streamlit\secrets.toml")["secrets"]
def get_query(query,med_class):
    index_path = "VectorStore/"+med_class
    print(index_path)
    embeddings = HuggingFaceInstructEmbeddings()
    docsearch = FAISS.load_local(index_path,embeddings)
    USER_ID = secrets["Llama_USER_ID"]
    APP_ID = secrets["Llama_APP_ID"]
    MODEL_ID = secrets["Llama_MODEL_ID"]
    PAT = secrets["PAT"]
    clarifai_llm = Clarifai(
    pat=PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
    )
    chain = load_qa_chain(clarifai_llm, chain_type="stuff")
    docs = docsearch.similarity_search(query,k=5)
    query_response = chain.run(input_documents=docs, question=query)
    return query_response
