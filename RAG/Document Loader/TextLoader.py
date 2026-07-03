from langchain_community.document_loaders import TextLoader

Data = TextLoader("B:\\AI\\RAG\\Document Loader\\LSTM.txt")
documents = Data.load()
print(documents)