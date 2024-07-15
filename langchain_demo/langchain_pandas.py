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

from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

if not os.environ.get("OPENAI_API_KEY"):
    import key
    key.init()
    assert os.environ.get('OPENAI_API_KEY')
    
############# Set Up DataFrame ############

path = './model_experiment_fields_ScenarioMIP_CMIP_filename_dates.csv' # path to CMIP6 datasets info
path_vars = '../../cmip6-cmor-tables/Tables' # path to directory of jsons describing CMIP6 variables

df = pd.read_csv(path)
df['collection'] = 'giss_cmip6'
df = df[['collection', 'MIP', 'model', 'experiment', 'variant', 'tableID', 'variable', 'grid', 'version', 'start_YM', 'end_YM', 'filename']]
df.columns = ['collection', 'MIP', 'model', 'experiment', 'variant', 'temporal resolution', 'variable', 'grid', 'version', 'start year-month', 'end year-month', 'filename']
df = df.astype(str)

def url(x): # add column with URL
    cols = '/'.join(x)
    return 'portal.nccs.nasa.gov/datashare/' + cols
df['URL'] = df.apply(lambda x: url(x), axis=1)

df['start year-month'] = df['start year-month'].apply(lambda x: x[:4]+'-'+x[4:])
df['end year-month'] = df['end year-month'].apply(lambda x: x[:4]+'-'+x[4:])

# remove version duplicates (keep just newest)
df = df.sort_values(df.columns.to_list(), ascending=True).drop_duplicates(
    subset=set(df.columns.to_list())-set(['version', 'filename', 'URL']),
    ignore_index=True, keep='last')

defs_temporal = {
    'AERmon': 'aerosols monthly (AERmon)',
    'Amon': 'atmospheric monthly (Amon)',
    'CFmon': 'cloud fraction monthly (CFmon)',
    'Emon': 'radiation monthly (Emon)',
    'EmonZ': 'EmonZ (EmonZ)',
    'LImon': 'land ice monthly (LImon)',
    'Lmon': 'land monthly (Lmon)',
    'Omon': 'ocean monthly (Omon)',
    'SImon': 'sea ice monthly (SImon)',
    '6hrLev': '6-hourly data on model levels (6hrLev)',
    '6hrPlev': '6-hourly data on pressure levels (6hrPlev)',
    '6hrPlevPt': '6-hourly data on pressure levels at point locations (6hrPlevPt)',
    'Eday': 'radiation daily (Eday)',
    '3hr': 'three (3) hour (3hr)',
    'CF3hr': 'CF three (3) hour (CF3hr)',
    'CFday': 'cloud fraction daily (CFday)',
    'E3hrPt': '3-hourly data at point locations (E3hrPt)',
    'day': 'daily (day)',
    'AERday': 'aerosols daily (AERday)',
    'SIday': 'sea ice daily (SIday)',
    'AERhr': 'aerosols hourly (AERhr)',
}

# read DataFrames from the var jsons, cat them all together
jdfs = []
for fname in os.listdir(path_vars):
    with open(os.path.join(path_vars, fname)) as j:
        d = json.load(j)
    
    try: 
        jdf = pd.DataFrame.from_dict(d['variable_entry'], orient='index')
    except:
        print('skipping', fname, 'due to formatting issue')
        continue    
    
    jdf['temporal resolution'] = fname[len('CMIP6_'):-len('.json')]
    jdfs.append(jdf)
    
varsdf = pd.concat(jdfs)

varsdf.rename(columns={'out_name': 'variable'}, inplace=True)
varsdf = varsdf.drop_duplicates(subset=['variable', 'temporal resolution'])

dfm = df.merge(varsdf, how='left')
dfm = dfm.drop_duplicates(subset=set(dfm.columns.to_list())-set(['dimensions']))
dfmsub = dfm.drop(['temporal resolution', 'grid', 'version', 'type', 'positive', 'valid_min', 'valid_max', 
        'ok_min_mean_abs', 'ok_max_mean_abs', 'flag_values', 'flag_meanings'], axis=1)


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
model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
chain_standalone = contextualize_q_prompt | model | StrOutputParser()

tool = PythonAstREPLTool(locals={"df": dfmsub})
llm_with_tools = model.bind_tools([tool], tool_choice=tool.name)
parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

system = f"""
You are a climate scientist with a pandas dataframe `df` that lists and describes all datasets 
within CMIP6. Here is the output of `df.head().to_markdown()`: \
{dfmsub.head().to_markdown()} \
Given a colleague's query, write and execute the Python code to find relevant CMIP6 datasets. \
Return ONLY the valid Python code and nothing else. Don't assume you have access to any libraries 
other than built-in Python ones, pandas, and scipy.
"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
pd_chain = prompt | llm_with_tools | parser | tool


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

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{prompt}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chain_reply = rag_prompt | model | StrOutputParser()

# standalone = chain_standalone.invoke(conversation)
# out = agent.invoke(standalone)
# df_ans = out['intermediate_steps'][0][1]
# chain_reply.invoke(conversation)

def predict(message, historystr, historychn):
    conversation = {'input': message, 'chat_history':historychn}
    standalone = chain_standalone.invoke(conversation) + '?' # sometimes an error if query isn't question-like enough
    df_ans = pd_chain.invoke(standalone)
    
    try:
        if df_ans.shape[0] > 10:
            df_ans = df_ans.iloc[:10, :]
    except:
        df_ans = None
        
    ai_message = chain_reply.invoke(dict(conversation, **{'prompt': get_rag_system_prompt(df_ans)}))

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