from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
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
retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model_name="mistral-small-2506")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
    )

Query = "What is LSTM? Explain in detail with example."
docs = multi_query_retriever.invoke(Query)

print("Retrieved Documents:")
for doc in docs:
    print(doc.page_content)
    

