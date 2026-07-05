from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("B:\\AI\\RAG\\deep-learning-material-dept-ece-ase-blr-1.pdf")
docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )

chunks = text_splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
    )

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="B:\\AI\\RAG\\Project_RAG\\DB"
    )
