# Init Chat Model method
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
api = os.getenv("MISTRAL_API_KEY")
if not api:
    raise ValueError("MISTRAL_API_KEY not found")
model = init_chat_model("mistral-small-2506", api_key=api)
result = model.invoke("What is the capital of France?")
print(result.content)


# Model Class method
import os
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
api = os.getenv("MISTRAL_API_KEY")
if not api:
    raise ValueError("MISTRAL_API_KEY not found")
model = ChatMistralAI(model="mistral-small-2506", mistral_api_key=api)
result = model.invoke("How many moons does Jupiter have?")
print(result.content)