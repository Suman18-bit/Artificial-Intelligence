from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # ✅ use new package
from langchain_chroma import Chroma                       # ✅ use new package
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")
# Vector store
vectorstore = Chroma(
    persist_directory="B:\\AI\\RAG\\Project_RAG\\DB",
    embedding_function=embeddings
)

# Retriever (MMR search)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Mistral LLM
llm = ChatMistralAI(model="mistral-small")

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Only use the information provided in the context to answer the question.

        If the answer is not in the context, respond with:
        "I don't know based on the given context."
        Do not invent or assume information.
        """
    ),
    (
        "human",
        """Context:
        {context}

        Question:
        {question}
        """
    )
])

print("Enter your question (or type '0' to quit):")

while True:
    question = input("\n\nYou:-> ")
    if question.lower() == "0":
        break

    # Retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build final prompt
    final_prompt = prompt_template.format(
        context=context,
        question=question
    )

    # Get answer
    result = llm.invoke(final_prompt)
    print("\n\nAI:-> ", result)
