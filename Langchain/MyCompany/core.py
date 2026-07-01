from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a professional assistant for movie information extraction.
your task is to answer questions about movies based on the information provided.

Output : 
movie title: {{title}},
release year: {{year}},
director: {{director}},
main actors: {{actors}},
genre: {{genre}},
and a brief summary of the movie.

If the information is not available, respond with "Information not available." """),

        ("human", "Please provide the paragraph: {paragraph}"),
    ]
)

pera = input("Please provide the paragraph: ")

final_prompt = prompt.invoke({"paragraph": pera})

model = ChatMistralAI(model="mistral-small-2506")

response = model.invoke(final_prompt)
print(response.content)