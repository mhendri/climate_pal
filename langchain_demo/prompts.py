from langchain_core.prompts import PromptTemplate

def get_split_prompt():
    template = '''
    You are a NASA climate scientist. Your colleague inquires about the CMIP6 dataset, asking for different datasets varying by variables of interest and temporal resolutions. 
    You must split your colleague's query into individual queries for each variable and temporal resolution pairs. If there is just one query then just give the single query.
    
    Here are some examples for clarification:
    
    Example 1:
    Original query: "I'm interested in surface temperature data on a monthly basis. I'd also like to see the data for aerosol on a daily basis."
    split query 1: "I'm interested in surface temperature data on a monthly basis."
    split query 2: "I'd  like to see the data for aerosol on a daily basis."
    
    Example 2:
    Original query: "I'm interested in surface temperature data on a monthly basis."
    split query 1: "I'm interested in surface temperature data on a monthly basis."
    
    Example 3:
    Original query: "I'd like to see the data for aerosol on a daily basis and precipitation on a 3 hour basis. I'm also interested in surface temperature data on a monthly basis."
    split query 1: "I'd like to see the data for aerosol on a daily basis."
    split query 2: "I'd like to see the data for precipitation on a 3 hour basis."
    split query 3: "I'm interested in surface temperature data on a monthly basis."
    
    Here is the original query:
    {input}
    '''
    return PromptTemplate(template=template, input_variables=["input"])


def get_adjustment_prompt():
    template = '''
    You are a NASA climate scientist. Your colleague inquires about the CMIP6 dataset, asking for different datasets varying by variables of interest and temporal resolutions.
    Your colleague has an initial list of queries but they need to be adjusted. You must adjust them.
    
    Initial queries:
    {initial_queries}
    
    Adjustment:
    {adjustment}
    '''
    
    return PromptTemplate(template=template, input_variables=["initial_queries", "adjustment"])
