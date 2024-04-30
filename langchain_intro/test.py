from langchain_intro.chatbot import chat_model, review_chain, hospital_agent_executor
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_prompts():
    system_template_str = """Your job is to use patient
                                reviews to answer questions about their experience at a
                                hospital. Use the following context to answer questions.
                                Be as detailed as possible, but don't make up any information
                                that's not from the context. If you don't know an answer, say
                                you don't know.

                                {context}
                            """

    user_template_str = """{question}"""

    system_prompt = SystemMessagePromptTemplate(prompt=
                                                PromptTemplate(template=system_template_str,
                                                               input_variables=["context"]))

    user_prompt = HumanMessagePromptTemplate(prompt=
                                             PromptTemplate(template=user_template_str,
                                                            input_variables=["question"]))

    messages = [system_prompt, user_prompt]

    prompt_template = ChatPromptTemplate(messages=messages, input_variables=["context", "question"])

    context = "The hospital was very clean and the staff was friendly."
    question = "What was the hospital like?"

    review_chain = prompt_template | chat_model

    response = review_chain.invoke({"context": context, "question": question})

    print(response)

def test_embedding_similarity_search():
    REVIEWS_CHROMA_PATH = "../chroma_data/"

    reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH, embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    question = "Has anyone complained about communication with the hospital staff?"

def test_retriever():
    question = "How was the hospital experience?"
    response = review_chain.invoke(question)

    print(response)

def test_agents():
    response = hospital_agent_executor.invoke({"input": "What is the wait time at hospital C?"})
    print(response)

    response = hospital_agent_executor.invoke({"input": "What is the wait time at hospital D?"})
    print(response)

    response = hospital_agent_executor.invoke({"input": "What is the general hospital experience?"})
    print(response)

if __name__ == "__main__":
    # test_embedding_similarity_search()
    # test_prompts()
    # test_retriever()
    test_agents()