#Init Chat Model method
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")
model = init_chat_model("gpt-4.1", api_key=api_key)
result = model.invoke("What is the capital of France?")
print(result.content)



#Model Class method 
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")
model = ChatOpenAI(model="gpt-4.1", api_key=api_key)
result = model.invoke("What is the capital of France?")
print(result.content)
