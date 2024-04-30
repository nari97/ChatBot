from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama


def get_retriever_chain():
    chat_model = ChatOllama(model="mistral")
    system_template_str = """
        START INSTRUCTIONS
        You are an assistant for question-answering tasks. You will be provided context to help answer the question.
        Each piece of context will also have the date the conversation happened. Use this information to help answer the question. 
        If you don't know the answer, just say that you don't know. Be as detailed as possible in your response.
        The context may contain extra information that is not relevant to the question. Use your judgment to filter out the relevant information and only return an answer that has utmost relevance to the question asked.
        END INSTRUCTIONS

        START CONTEXT
        {context}
        END CONTEXT
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

    conversation_vector_chain = prompt_template | chat_model

    return conversation_vector_chain


