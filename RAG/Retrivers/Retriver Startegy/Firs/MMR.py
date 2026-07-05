from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

data = TextLoader("B:\\AI\\RAG\\VectorDB\\LSTM.txt")
docs = data.load()

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_texts(
    texts=[doc.page_content for doc in docs],
    embedding=embeddings,
    persist_directory="B:\\AI\\RAG\\Retrivers\\Retriver Startegy\\DB_MMR"
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

print("MMR Retriever Results:")

mmr_docs = mmr_retriever.invoke("What is LSTM?")

for doc in mmr_docs:
    print(doc.page_content)
    print("-----")



