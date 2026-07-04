from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

Data = PyPDFLoader("B:\\AI\\RAG\\Text Spliter\\deep-learning-material-dept-ece-ase-blr-1.pdf")
documents = Data.load()

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

print(len(chunks))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("-" * 50)
