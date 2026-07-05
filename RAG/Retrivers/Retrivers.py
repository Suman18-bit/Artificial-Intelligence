from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

data = TextLoader("B:\\AI\\RAG\\VectorDB\\LSTM.txt")
docs = data.load()

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma.from_texts(
    texts=[doc.page_content for doc in docs],
    embedding=embeddings,
    persist_directory="B:\\AI\\RAG\\Retrivers\\ChromaDB2"
)

result = vectorstore.similarity_search("What is the purpose of LSTM?", k=3)

for r in result:
    print(r.page_content)
    print(r.metadata)


retrievers = vectorstore.as_retriever()

docs = retrievers.invoke("What is the purpose of LSTM?")

for d in docs:
    print(d.page_content)
    print(d.metadata)