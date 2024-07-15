import os
from langchain_openai import ChatOpenAI

def llm_init(model_name="gpt-3.5-turbo",
             temperature=0,
             max_tokens=4096,
             model_kwargs={'frequency_penalty':0, 'presence_penalty':0}):

    llm = ChatOpenAI(model_name=model_name,
                     temperature=temperature,
                     max_tokens=max_tokens,
                     model_kwargs=model_kwargs,
                     openai_api_key=os.getenv('OPENAI_API_KEY'))

    return llm