# Configuration
pdagent_type = 'auto'  # Options: 'auto', 'manual'
llm_type = 'gpt-3.5-turbo'  # Options: 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-4o'
initialization_params = {}  # JSON version

# Imports
import os
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.tools import tool

# Ensure OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    import key
    key.init()
assert os.environ.get('OPENAI_API_KEY')

# Load and process data
path = './model_experiment_fields_ScenarioMIP_CMIP_filename_dates.csv'
df = pd.read_csv(path)
df['collection'] = 'giss_cmip6'
df['org'] = 'NASA-GISS'
df = df[['collection', 'MIP', 'org', 'model', 'experiment', 'variant', 
         'tableID', 'variable', 'grid', 'version', 'start_YM', 'end_YM', 'filename']]
df.columns = ['collection', 'MIP', 'org', 'model', 'experiment', 'variant', 
              'temporal resolution', 'variable', 'grid', 'version', 'start year', 'end year', 'filename']
df = df.astype(str)

# Generate URLs
def generate_url(x):
    cols = '/'.join([val for val in x])
    return 'https://portal.nccs.nasa.gov/datashare/' + cols

url_col_names = df.columns[:-3].to_list() + ['filename']
urldf = df[url_col_names]
df['URL'] = urldf.apply(lambda x: generate_url(x), axis=1)

# Convert year columns to integers
df['start year'] = df['start year'].apply(lambda x: int(x[:4]))
df['end year'] = df['end year'].apply(lambda x: int(x[:4]))

# Remove old-version datasets
df = df.sort_values(df.columns.to_list(), ascending=True).drop_duplicates(
    subset=set(df.columns.to_list()) - set(['version', 'filename', 'URL']),
    ignore_index=True, keep='last')

# Clean temporal resolution
def clean_resolution(reso):
    resos = ['hr', 'day', 'mon']
    for q in resos:
        if q in reso:
            return q
    return 'NA'

df['temporal resolution'] = df.apply(lambda x: clean_resolution(x['temporal resolution']), axis=1)

# Load variable information
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
varsdf['extended_comment'] = varsdf.apply(lambda x: f"{x['long_name']} ({x['variable']}): {x['comment']}", axis=1)

varsdf = varsdf[varsdf['variable'].isin(df['variable'].unique())]

# Query summarizer agent
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

# Variables agent
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
varsdf.loc[varsdf['extended_comment'].isna(), 'extended_comment'] = ''
varsdf['embeds'] = list(np.asarray(embeddings.embed_documents(varsdf['extended_comment'])))

@tool
def find_relevant_variables(keywords):
    """Search dataframe of CMIP6 variables to find relevant variables for a list of keywords."""
    if isinstance(keywords, list):
        keywords = ' '.join(keywords)
    emb = np.asarray([embeddings.embed_query(keywords)])
    docslist = varsdf['embeds'].to_list()
    docslist.extend(emb)
    sims = cosine_similarity(np.stack(docslist))
    sims = sims[-1, :-1]  # Row -1 is the keywords, all other rows are the variable descriptors
    simssort = np.argsort(sims)[-5:][::-1]
    dfmatch = deepcopy(varsdf.iloc[simssort, :])
    dfmatch.loc[:, 'score'] = sims[simssort]
    return dfmatch

llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm_with_tools = llm.bind_tools([find_relevant_variables], tool_choice=find_relevant_variables.name)
parser = JsonOutputKeyToolsParser(key_name=find_relevant_variables.name, first_tool_only=True)

system = """
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

# Temporal frequency agent
temporalmsg = """You are an expert climate scientist. Is the following CMIP6-related query 
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

# Year range agent
yearmsg = """You are an expert climate scientist. Does the following CMIP6 query require or specify 
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
    start = int(start) if start != 'NA' else float('Inf')
    end = int(end) if end != 'NA' else -1 * float('Inf')
    return start, end

# Function to get dataset URL
def get_dataset_url(query):
    conversation = {'input': query, 'chat_history': []}
    standalone = chain_standalone.invoke(conversation)
    
    # Find relevant variables
    variables = varagent.invoke(query)['variable'].to_list()
    
    # Determine temporal resolution
    temporal = temporalagentparsed(query)
    
    # Determine year range
    startyr, endyr = yearagentparsed(standalone)
    
    # Find the best matching dataset URL
    urldf = df[(df['variable'].isin(variables)) & 
               (df['temporal resolution'] == temporal) & 
               (df['start year'] <= startyr) & 
               (endyr <= df['end year'])].sort_values('variable', 
                                                      key=lambda s: s.apply(variables.index), 
                                                      ignore_index=True)
    if len(urldf) > 0:
        return urldf.loc[0, 'URL']
    else:
        return "No matching dataset found."

# Example usage
query = "Plot the 10, 20, 50, and 100 yr return period for maximum daily rainfall based on historical simulations -> Plot return periods for maximum daily rainfall"
url = get_dataset_url(query)
print(url)


#https://portal.nccs.nasa.gov/datashare/giss_cmip6/ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp585/r1i1p1f2/day/tasmax/gn/v20200115/tasmax_day_GISS-E2-1-G_ssp585_r1i1p1f2_gn_20150101-21001231.nc


#https://portal.nccs.nasa.gov/datashare/giss_cmip6/ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp585/r1i1p1f2/day/sfcWindmax/gn/v20200115/sfcWindmax_day_GISS-E2-1-G_ssp585_r1i1p1f2_gn_24510101-25001231.nc