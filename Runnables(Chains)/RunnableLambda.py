from dotenv import load_dotenv
load_dotenv()  

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

model = ChatMistralAI(model="mistral-small-2506")

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in detail.")
])

parser = StrOutputParser()

Parallel = RunnableParallel({
    "Short": RunnableLambda(lambda x :x ['Short']) | short_prompt | model | parser,
    "Detailed": RunnableLambda(lambda x :x ['Detailed']) | detailed_prompt | model | parser
})

response = Parallel.invoke({
    "Short":{"topic": "AI"},
    "Detailed":{"topic": "Deep Learning"}
                           })


print("The Short Answer:",response['Short'])
print("-"*100)
print("The Detailed Answer:",response['Detailed'])
