from dotenv import load_dotenv
load_dotenv()  

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Explain{topic} in simple word"
)

model = ChatMistralAI(model="mistral-small-2506")

parser = StrOutputParser()

formatted_prompt = prompt.format_messages(topic="Machine Learning")

result = model.invoke(formatted_prompt)

final_result = parser.parse(result.content)

print(final_result)