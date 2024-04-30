from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama


def get_er_agent():
    system_template_str = open("../prompts/entity_recognition_system_prompt.txt", "r").read()
    user_template_str = open("../prompts/entity_recognition_user_prompt.txt", "r").read()

    system_prompt = SystemMessagePromptTemplate.from_template(template=system_template_str)
    user_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(template=user_template_str, input_variables=["summary"]))

    messages = [system_prompt, user_prompt]

    prompt_template = ChatPromptTemplate(messages=messages, input_variables=["summary"])

    chat_model = ChatOllama(model="mistral")
    er_agent = prompt_template | chat_model

    return er_agent
