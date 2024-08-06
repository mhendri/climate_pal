# config
pdagent_type = 'auto' #one of: auto, manual
llm_type = 'gpt-3.5-turbo' #one of: gpt-3.5-turbo-0613, gpt-3.5-turbo-0125, gpt-3.5-turbo, gpt-4o
initialization_params = {} #eg json version

import pandas as pd
import os
import json
import openai
import numpy as np

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
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.tools import tool
from copy import deepcopy

def parse_variables(string): #since some gold standard answers contain more than one right variable
    if ',' in string:
        return set(string.split(','))
    else:
        return string

if not os.environ.get("OPENAI_API_KEY"):
    import key
    key.init()
assert os.environ.get('OPENAI_API_KEY')
    
path = './model_experiment_fields_ScenarioMIP_CMIP_filename_dates.csv'
df = pd.read_csv(path)
df['collection'] = 'giss_cmip6'
df['org'] = 'NASA-GISS'
df = df[['collection', 'MIP', 'org', 'model', 'experiment', 'variant', # reorder columns
         'tableID', 'variable', 'grid', 'version', 'start_YM', 'end_YM', 'filename']]
df.columns = ['collection', 'MIP', 'org', 'model', 'experiment', 'variant', # rename columns
         'temporal resolution', 'variable', 'grid', 'version', 'start year', 'end year', 'filename']
df = df.astype(str)

# https://portal.nccs.nasa.gov/datashare/giss_cmip6/ScenarioMIP/NASA-GISS/GISS-E2-1-G/
# ssp534-over/r1i1p3f1/Amon/rsdt/gn/v20200115/
# rsdt_Amon_GISS-E2-1-G_ssp534-over_r1i1p3f1_gn_204001-210012.nc

url_col_names = df.columns[:-3].to_list() + ['filename'] # removes start/end year, filename
urldf = df[url_col_names] # working off this extra dataframe runs like 5x faster
def url(x):
    cols = '/'.join([val for val in x])
    return 'https://portal.nccs.nasa.gov/datashare/' + cols
df['URL'] = urldf.apply(lambda x: url(x), axis=1)

df['start year'] = df['start year'].apply(lambda x: int(x[:4])) #+'-'+x[4:])
df['end year'] = df['end year'].apply(lambda x: int(x[:4])) #+'-'+x[4:])

oldn = df.shape[0]
# keep only the latest version of each dataset
df = df.sort_values(df.columns.to_list(), ascending=True).drop_duplicates(
    subset=set(df.columns.to_list())-set(['version', 'filename', 'URL']),
    ignore_index=True, keep='last')
print('removed', oldn-df.shape[0], 'rows corresponding to old-version datasets')

# go from "Amon"->"mon", for instance
def clean_resolution(reso):
    resos = ['hr', 'day', 'mon']
    for q in resos:
        if q in reso:
            return q
    return 'NA'
df['temporal resolution'] = df.apply(lambda x: clean_resolution(x['temporal resolution']), axis=1)

#read in info on the variables in CMIP6
path = '../../cmip6-cmor-tables/Tables'
jdfs = []
for fname in os.listdir(path):
    with open(os.path.join(path, fname)) as j:
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
varsdf = varsdf.loc[:, ['long_name', 'comment', 'variable']]
varsdf.drop_duplicates(inplace=True)
varsdf['extended_comment'] = varsdf.apply(lambda x: str(x['long_name'])+' ('+str(x['variable'])+'): ' + str(x['comment']), axis=1)

varsdf = varsdf[varsdf['variable'].isin(df['variable'].unique())] # keep only variables in 
df = df[df['variable'].isin(varsdf['variable'])]
print('varsdf has', len(varsdf), 'rows and datasets df has', len(df), 'rows')

# query summarizer agent
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
model = ChatOpenAI(model=llm_type, temperature=0)
chain_standalone = contextualize_q_prompt | model | StrOutputParser()

#variabless agent
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
varsdf.loc[varsdf['extended_comment'].isna(), 'extended_comment'] = ''

embs = np.asarray(embeddings.embed_documents(varsdf['extended_comment']))
varsdf['embeds'] = list(embs) 

@tool 
def find_relevant_variables(keywords):
    """search dataframe of CMIP6 variables, to find relevant variables to a list of keywords"""
    if type(keywords) == list:
        keywords = ' '.join(keywords)
    # print(keywords)
    emb = np.asarray([embeddings.embed_query(keywords)])
    docslist = varsdf['embeds'].to_list()
    docslist.extend(emb)
    sims = cosine_similarity(np.stack(docslist))
    sims = sims[-1, :-1]  # row -1 is the keywords, all other rows are the variable descriptors
    simssort = np.argsort(sims)[-5:][::-1]
    dfmatch = deepcopy(varsdf.iloc[simssort,:])
    dfmatch.loc[:,'score'] = sims[simssort]
    return dfmatch

# find_relevant_variables('maximum temperature')

llm=ChatOpenAI(temperature=0, model="gpt-4o", )

llm_with_tools = llm.bind_tools([find_relevant_variables], tool_choice=find_relevant_variables.name)
parser = JsonOutputKeyToolsParser(key_name=find_relevant_variables.name, first_tool_only=True)

system = f"""
You are a climate scientist and expert on the CMIP6 dataset. Given a colleague's query, 
find CMIP6 **variables** most likely to help answer the query. To find these variables, 
1. summarize the variable-related keywords in the query (e.g. "rain", not analysis words like "plot") 
using words that are useful for describing the CMIP 6 datasets. For instance, instead of "weather month", say 
"temperature precipitation wind", because "month" is not relevant to the variable choice and 
"temperature precipitation wind" is more specific than "weather". If the query relates to whether a 
variable surpasses some threshold, you may wish to search for the "min" or "max" versions of variables.
2. connect your list of summary words into one string.
3. pass the keywords string to the provided tool and return the result.
"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
varagent = prompt | llm_with_tools | parser | find_relevant_variables

# conversation = {'input': "Average annual days surface air temperature exceeds 98°F in 2050-2070",
#                 'chat_history': []}
# standalone = chain_standalone.invoke(conversation)
# print(standalone)
# varagent.invoke(standalone)

# temporal frequency agent
temporalmsg = f"""You are an expert climate scientist. Is the following CMIP6-related query 
best answered using data gathered at which of the following resolutions? 
A. hour
B. day
C. month
D. not applicable, none of the above, or unclear
Respond with only the one letter corresponding to your choice and nothing else. If a query does 
not specify any given temporal resolution, like the query "plot average temperature", then choose 
option D.
"""
temporalprompt = ChatPromptTemplate.from_messages([('system', temporalmsg), ('human', '{question}')])
temporalagent = temporalprompt | llm 
key = {'A': 'hr', 'B': 'day', 'C': 'mon', 'D': 'NA'}

def temporalagentparsed(query):
    pred = temporalagent.invoke(query).content
    return key.get(pred[0], 'NA')

# temporalagent.invoke(' frost data')

# year range agent
yearmsg = f"""You are an expert climate scientist. Does the following CMIP6 query require or specify 
a year range for the data required to answer the query? If yes, provide the year range in format 
START-END, for instance 1960-1970 or 2100-3100. If no, respond NA-NA. If only the start or end is 
specified, provide just that year in format START-NA (eg 2100-NA) or NA-END (eg NA-1900).
Provide only the year range in this format and nothing else.
"""
yearprompt = ChatPromptTemplate.from_messages([('system', yearmsg), ('human', '{question}')])
yearagent = yearprompt | llm

def yearagentparsed(standalone):
    string = yearagent.invoke(standalone).content
    start, end = string.split('-')
    if start != 'NA':
        start = int(start)
    else: 
        start = float('Inf')
    if end != 'NA':
        end = int(end)
    else:
        end = -1*float('Inf')
    return start, end

# yearagentparsed('average temperature in 20th century')
queries = pd.read_excel('evals.xlsx')
queries.loc[queries['start year'].isna(), 'start year'] = float('Inf')
queries.loc[queries['end year'].isna(),   'end year'] = -1 * float('Inf')
queries.loc[queries['frequency'].isna(),  'frequency'] = 'NA'
queries.head()

# %%capture
outs = []
score = 0
yrscore = 0
resscore = 0
for i, r in queries.iterrows():
    qscore = 0
    yrqscore = 0
    resqscore = 0

    conversation = {'input': r['query'], 'chat_history': []}
    standalone = chain_standalone.invoke(conversation)
    print(r['query'], '->', standalone)
    
    variables = varagent.invoke(r['query'])['variable'].to_list()
    # variablesgold = parse_variables(r['variable'])
    variablesgold = r['variable']
    # print(variables)
    if ', etc' in variablesgold: #any answer correct?
        qscore += 1
    elif len(variablesgold) > 1: #more than one right answer
        if set.intersection(set(variables), set(variablesgold.split(', '))):
            qscore += 1
    else: # just one correct answer
        if variablesgold in variables:
            qscore += 1
    print('\tvariable question score', qscore)
            
    # print(response, variables, qscore)
    
    temporal = temporalagentparsed(r['query'])
    if temporal == r['frequency']:
        resqscore = 1
    print('\ttemporal question score', resqscore, temporal, r['frequency'])
    
    startyr, endyr = yearagentparsed(standalone)
    # print(startyr, endyr, r['start year'], r['end year'])
    if r['start year'] <= startyr:
        yrqscore += 0.5
    if endyr <= r['end year']:
        yrqscore += 0.5
    print('\tyear question score', yrqscore)
    
    score += qscore
    yrscore += yrqscore
    resscore += resqscore
    
    # choose dataset
    urldf = df[df['variable'].isin(variables) & (df['temporal resolution'] == temporal) & \
        (df['start year'] <= startyr) & (endyr <= df['end year'])].sort_values('variable', 
        key=lambda s: s.apply(variables.index), ignore_index=True)
    if len(urldf) > 0:
        url = urldf.loc[0, 'URL'] #sort so highest-similarity variable is first
        # https://stackoverflow.com/questions/52784410/sort-column-in-pandas-dataframe-by-specific-order
        print(url)
    
score = score / len(queries)
yrscore = yrscore / len(queries)
resscore = resscore / len(queries)
print('variable score', score)
print('year score', yrscore)
print('temporal resolution score', resscore)