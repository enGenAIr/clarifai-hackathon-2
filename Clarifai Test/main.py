from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Clarifai
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR



embeddings = HuggingFaceInstructEmbeddings()
docsearch = FAISS.load_local("c-retard",embeddings)

USER_ID = "meta"
APP_ID = "Llama-2"
MODEL_ID = "llama2-70b-chat"
clarifai_llm = Clarifai(
    pat="", user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
)


chain = load_qa_chain(clarifai_llm, chain_type="stuff")


query = "What Precautions should I take"
docs = docsearch.similarity_search(query,k=2,)
x = chain.run(input_documents=docs, question=query)
print(x)