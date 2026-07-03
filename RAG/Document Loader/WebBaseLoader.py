from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/mac-studio/"

Data = WebBaseLoader(url)

documents = Data.load()
print(documents[0].page_content) 