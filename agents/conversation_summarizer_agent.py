import langchain_community.chat_models
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate


def get_summarizer():
    chat_model = langchain_community.chat_models.ChatOllama(model="mistral")

    system_template_str = """
    START INSTRUCTIONS
    You will be provided with a conversation. Your job is to summarize the conversation and also detect the sentiment for the conversation. The options for sentiment are positive, negative or neutral. 
    The conversations are extracted from a group that talks about the sport of soccer. There will be a lot of soccer(football) jargon in the conversation. There will also be jargon related to Fantasy Premier League, which is a fantasy football game. Use your knowledge to decipher players, teams and other football-related terms.
    Be as detailed as possible when making the summary, and don't make up any information that's not from the conversation. If you don't know an answer, say you don't know. Use the football jargon in the conversation to make the summary as accurate as possible.
    In the description, try to highlight the people involved in the conversation instead of using generic terms like "they" or "them". Follow the same principle if you are talking about the teams or the players. Try to name them instead of using generic terms.
    Do not use explicit language in your response. Do not talk about politics or religion. Do not make any assumptions about the people involved in the conversation. Do not mention anything sexual or violent. Do not use any offensive language.
    Do not include any instance of explicit language in your response.
    Each message in the conversation will contain the name of the sender and the message they sent. The conversation will be provided in the following format:
    Sender1: Message1
    Sender2: Message2
    
    Your output should be in a JSON format with two keys:
    "Summary": The summary of the conversation, 
    "Sentiment": The sentiment of the conversation (positive, negative or neutral, all lower case)
    END INSTRUCTIONS
    
    """

    user_template_str = """
    START CONTEXT
    {conversation}
    END CONTEXT"""

    system_prompt = SystemMessagePromptTemplate.from_template(template=system_template_str)

    user_prompt = HumanMessagePromptTemplate(prompt=
                                             PromptTemplate(template=user_template_str,
                                                            input_variables=["conversation"]))

    messages = [system_prompt, user_prompt]

    prompt_template = ChatPromptTemplate(messages=messages, input_variables=["conversation"])

    summary_chain = prompt_template | chat_model

    return summary_chain


