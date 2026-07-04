from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=1
    )

Data = PyPDFLoader("B:\\AI\\RAG\\Text Spliter\\deep-learning-material-dept-ece-ase-blr-1.pdf")

documents = Data.load()

splits = splitter.split_documents(documents)

print(len(splits))

Data2 = TextLoader("B:\\AI\\RAG\\Text Spliter\\LSTM.txt")

documents = Data2.load()

splitter2 = CharacterTextSplitter(
    separator="",
    chunk_size=1000,
    chunk_overlap=1
    )

splits2 = splitter2.split_documents(documents)

print(len(splits2))

for i, split in enumerate(splits2):
    print(f"Split {i+1}:")
    print(split.page_content)
    print("-" * 50)

for i, split in enumerate(splits):
    print(f"Split {i+1}:")
    print(split.page_content)
    print("-" * 50)






