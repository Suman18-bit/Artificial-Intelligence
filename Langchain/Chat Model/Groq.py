# Init Chat Model method
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
api = os.getenv("GROQ_API_KEY")
if not api:
    raise ValueError("GROQ_API_KEY not found")
model = init_chat_model("groq/compound-mini", api_key=api)
result = model.invoke("What is the capital of France?")
print(result.content)


# Model Class method
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
api = os.getenv("GROQ_API_KEY")
if not api:
    raise ValueError("GROQ_API_KEY not found")
model = ChatGroq(model="groq/compound-mini", groq_api_key=api)
result = model.invoke("What is the capital of France?")
print(result.content)