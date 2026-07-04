from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

#PDF file path

Data = PyPDFLoader("B:\\AI\\RAG\\Text Spliter\\deep-learning-material-dept-ece-ase-blr-1.pdf")
documents = Data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

print(len(chunks))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("-" * 50)


#Text file path

Data2 = TextLoader("B:\\AI\\RAG\\Text Spliter\\LSTM.txt")
documents2 = Data2.load()

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks2 = splitter2.split_documents(documents2)

print(len(chunks2))

for i, chunk in enumerate(chunks2):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("-" * 50)