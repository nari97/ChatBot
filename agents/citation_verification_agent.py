from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatOllama


def get_agent():
    chat_model = ChatOllama(model="mistral")

    system_template_str = """
    START INSTRUCTIONS
    You are a citation and verification assistant. You will be provided three inputs: a question, a claim that answers the question and the context for the answer.
    Your job is to edit the claim if it is not accurate with respect to the question and context. You will also add citation links to the claim where necessary.
    The context will contain information that can be used to verify the claim. Use this information to edit the claim verifying that it answers the question and add citations.
    For your output, you will provide the edited claim with citations in the format [Source ID]. Strictly adhere to the output description and do not return any other information. Do not make up any information that's not from the context. If you don't know an answer, say you don't know.
    Example:
        START CONTEXT
            Question: Who does the group think is the best manager for Manchester United?
            Claim: The group thinks that Erik Ten Hag is the best manager for Manchester United. Erik Ten Hag is a Dutch football manager who is currently the head coach of Ajax.
            Context:
            Source ID: 7113
            Source summary: The group is discussing the best manager for Manchester United. They are considering Erik Ten Hag as a potential candidate.
            
            Source ID: 4352
            Source summary: Erik Ten Hag is a Dutch football manager who is currently the head coach of Ajax.
            
        END CONTEXT
        
        Output: The group thinks that Erik Ten Hag is the best manager for Manchester United [7113].
        
    END INSTRUCTIONS
    """

    user_template_str = """
    START CONTEXT
    {context}
    END CONTEXT
    
    Output: 
    """

    system_prompt = SystemMessagePromptTemplate.from_template(template=system_template_str)

    user_prompt = HumanMessagePromptTemplate(prompt=
                                             PromptTemplate(template=user_template_str,
                                                            input_variables=["context"]))

    messages = [system_prompt, user_prompt]

    prompt_template = ChatPromptTemplate(messages=messages, input_variables=["context"])

    agent = prompt_template | chat_model

    return agent
