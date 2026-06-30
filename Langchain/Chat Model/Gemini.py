# Init Chat Model method
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
api = os.getenv("GEMINI_API_KEY")
if not api:
    raise ValueError("GEMINI_API_KEY not found")
model = init_chat_model("gemini-pro", api_key=api)

result = model.invoke("What is the capital of France?")
print(result.content)


# Model Class method
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
api = os.getenv("GEMINI_API_KEY")
if not api:
    raise ValueError("GEMINI_API_KEY not found")
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api)
result = model.invoke("What is the capital of France?")
print(result.content)