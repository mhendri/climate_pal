import openai
import chromadb
import gradio as gr
import os
import json
import pandas as pd

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

if not os.environ.get("OPENAI_API_KEY"):
    import key
    key.init()
    assert os.environ.get('OPENAI_API_KEY')

path = './paths.txt'
dicts = {}
with open(path, 'r') as file:
    for l in file.readlines():
        if '/v20' in l and '.html' in l:
            url = l[:-1] #remove newline char
            lsplit = url.split('/')
            url_clean = '/'.join(lsplit[:-1]) # so they go up to and including the version directory
            dicts[url_clean] = {'activity_id':lsplit[3],
                            'experiment_id':lsplit[6],
                            'temporal_resolution':lsplit[8],
                            'variable':lsplit[9]}
df = pd.DataFrame.from_dict(dicts, orient='index')

docsVAR= {
    'ta': 'air temperature. air temperature is the temperature in the atmosphere. It has units of Kelvin (K). Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15. This parameter is available on multiple levels through the atmosphere',
    'tas': 'air temperature near surface. air temperature near surface is the temperature of air at 2 meters above the surface of land, sea or inland waters',
    'pr': 'precipitation flux. precipitation flux is the flux of water equivalent (rain or snow) reaching the land surface. This includes the liquid and solid phases of water'
}
docsTR = {
    'Amon': 'monthly resolution, for atmospheric variables',
    '3hr': 'three (3) hour resolution',
    'day': 'daily resolution, measured every day',
    'AERmon': 'monthly resolution, for aerosol variables',
}
df['def_variable']             = df['variable'].apply(lambda x: docsVAR.get(x, 'UNK'))
df['def_temporal_resolution'] = df['temporal_resolution'].apply(lambda x: docsTR.get(x, 'UNK'))

dfsub = df[(df['def_temporal_resolution']!='UNK') & (df['def_variable']!='UNK')]


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "summarize the user's request in a few key words without punctuation. "
    "Do NOT attempt to answer the user's request. "
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
chain_standalone = contextualize_q_prompt | model | StrOutputParser()

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    dfsub,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    return_intermediate_steps=True
)


def get_rag_system_prompt(df_ans=None):
    if df_ans is None:
        rag_system_prompt = """
            Given a chat history and the latest user query, which may reference context
            in the chat history, answer the user.
            """
    else:
        rag_system_prompt = f"""
            You have access to a dataframe `df`. Here is the output of `df.to_markdown()`: 
            {df_ans.to_markdown()}

            Given a chat history and the latest user query, which may reference context
            in the chat history, use information in DataFrame `df` to best answer the user.
        """
    return rag_system_prompt


# standalone = chain_standalone.invoke(conversation)
# out = agent.invoke(standalone)
# df_ans = out['intermediate_steps'][0][1]
# chain_reply.invoke(conversation)

def predict(message, historystr, historychn):
    
    conversation = {'input': message, 'chat_history':historychn}
    standalone = chain_standalone.invoke(conversation)
    out = agent.invoke(standalone)
    
    try:
        df_ans = out['intermediate_steps'][0][1]
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_rag_system_prompt(df_ans)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    except:
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_rag_system_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    chain_reply = rag_prompt | model | StrOutputParser()
    ai_message = chain_reply.invoke(conversation)

    print(ai_message)
    historystr.append(
        (message, ai_message)
    )
    historychn.extend(
        [
            HumanMessage(content=message),
            AIMessage(content=ai_message)
        ]
    )

    return [historystr, historystr, historychn]

# Create the Gradio interface with streaming
def chat_interface_streaming():
    with gr.Blocks() as demo:
        chat = gr.Chatbot()
        msgs_str = gr.State([])
        msgs_chn = gr.State([])
        
        def respond(message, msgs_str, msgs_chn):
            msgs_str, new_state, msgs_chn = predict(message, msgs_str, msgs_chn)
            return msgs_str, new_state, msgs_chn
        
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        
        txt.submit(respond, [txt, msgs_str, msgs_chn], [chat, msgs_str, msgs_chn], queue=True)
    
    demo.launch(share=True)

# Launch the interface
# this breaks on NASA machines
chat_interface_streaming()

# Alternatively, invoke retriever with one off query
# print(retriever.invoke("What's one variable about snow with frequency mon"))
# print(self_query_retriever.invoke("I am interested in air temperature"))