from dotenv import load_dotenv
load_dotenv()  

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

parser = StrOutputParser()

final_result = LLMChain(
    llm=model,
    prompt=prompt,
    output_parser=parser
)

response = final_result.invoke({"topic": "machine learning"})
print(response)