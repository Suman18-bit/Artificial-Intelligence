from langchain_community.document_loaders import PyPDFLoader

Data = PyPDFLoader("B:\\AI\\RAG\\Document Loader\\LSTM_and_GRU_Notes.pdf")

documents = Data.load()
print(len(documents))
 #----------> Make each Page as a separate document, LSTM_and_GRU_Notes.pdf has 3 pages so the output should be 3