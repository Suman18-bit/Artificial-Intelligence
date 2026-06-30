import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

api = os.getenv("HUGGINGFACEHUB_API_KEY")
if not api:
    raise ValueError("HUGGINGFACEHUB_API_KEY not found")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=api 
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Hello, how are you?")
print(response.content)
