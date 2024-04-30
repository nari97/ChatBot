from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama


def get_timeframe_agent():
    system_template_str = """
    You are an assistant for question-answering tasks. You are the first model in a chain of models that will be used to answer the question.
    Given the question, your task is to detect and split the question if there is a time frame in the question. If there is no time frame, you should return the original question.
    The time frame can be between a set of dates, a specific date, or even something general like "last week" or "last 6 months".
    Assume that today's year is 2024, month is 04 and day is 01. If the question contains a time frame, you should return the question without the time frame.
    For your output, you will return the question without the time frame, and will return two additional pieces of information:
    timeframe_start: The start date of the time frame in the format "YYYY-MM-DD". If the time frame is not specific, you should return "2018-01-01"
    timeframe_end: The end date of the time frame in the format "YYYY-MM-DD". If the time frame is not specific, you should return "2024-04-01"
    
    Examples:
    # What are Nikhil's opinions on haaland in the last 6 months?
    Question: What are Nikhil's opinions on haaland?
    timeframe_start: 2023-10-01
    timeframe_end: 2024-04-01
    
    # What is the group's take on football?
    Question: What is the group's take on football?
    timeframe_start: 2018-01-01
    timeframe_end: 2024-04-01
    
    # Can you tell me what are Ajay's opinions on Bayern Munich between 2022 and 2023?
    Question: Can you tell me what are Ajay's opinions on Bayern Munich?
    timeframe_start: 2022-01-01
    timeframe_end: 2023-12-31
    
    # What did Akshay think about the match on 2024-03-01?
    Question: What did Akshay think about the match?
    timeframe_start: 2024-03-01
    timeframe_end: 2024-04-01
    """

    user_template_str = """Input question: {question}"""

    system_prompt = SystemMessagePromptTemplate.from_template(template=system_template_str)

    user_prompt = HumanMessagePromptTemplate(prompt=
                                             PromptTemplate(template=user_template_str,
                                                            input_variables=["question"]))

    messages = [system_prompt, user_prompt]

    prompt_template = ChatPromptTemplate(messages=messages, input_variables=["question"])

    chat_model = ChatOllama(model="mistral")

    agent = prompt_template | chat_model

    return agent
