import langchain_community.chat_models
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_json_chat_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from langchain_intro.tools import get_current_wait_time


chat_model = langchain_community.chat_models.ChatOllama(model="mistral")

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

REVIEWS_CHROMA_PATH = "../chroma_data/"

reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH, embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

reviewers_retriver = reviews_vector_db.as_retriever(k=10)

review_chain =  ({"context": reviewers_retriver, "question": RunnablePassthrough()} | prompt_template | chat_model)

tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

hospital_agent_prompt = hub.pull("hwchase17/react-chat-json")

agent_chat_model = langchain_community.chat_models.ChatOllama(model="mistral")

hospital_agent = create_json_chat_agent(
    llm=agent_chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)