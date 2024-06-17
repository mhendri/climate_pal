import openai
import chromadb
import gradio as gr
import os
import json

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = (
    "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"  # Replace with your key
)

with open("custom_cmor_tables_augmented/CMIP6_Amon_custom_20240110.json", "r") as file:
    data = json.load(file)

formatted_data = [
    Document(page_content=value["long_name"], metadata=value)
    for key, value in data["variable_entry"].items()
]
vectorstore = Chroma.from_documents(
    formatted_data,
    OpenAIEmbeddings(
        openai_api_key=""
    ),
)

metadata_field_info = [
    AttributeInfo(
        name="frequency",
        description="The frequency of the data",
        type="string",
    ),
    AttributeInfo(
        name="modeling_realm",
        description="The modeling realm of the data",
        type="string",
    ),
    AttributeInfo(
        name="standard_name",
        description="The standard name of the data",
        type="string",
    ),
    AttributeInfo(
        name="units",
        description="The units in which the data is measured",
        type="string",
    ),
    AttributeInfo(
        name="long_name",
        description="A descriptive long name for the data",
        type="string",
    ),
    AttributeInfo(
        name="comment",
        description="Additional comments or information about the data",
        type="string",
    ),
    AttributeInfo(
        name="augmented_comment",
        description="Additional comments or information about the data",
        type="string",
    ),
]

document_content_description = "climate diagnostic variables"

llm = ChatOpenAI(
    temperature=0, openai_api_key="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"
)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)


# Create a chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=self_query_retriever,
    return_source_documents=True,
)


# print(self_query_retriever.invoke("I am interested in one variable about snow and mon frequency")[0].page_content)

question = "what variables are available for snow"
llm_response = qa_chain({"query": question})
print(llm_response)


### Block 1
# works ok - recalls chat history and can retrieve variables from self query retreiver

# def predict(message, history):
#     # Convert the history to Langchain format
#     history_langchain_format = []
#     for human, ai in history:
#         history_langchain_format.append(HumanMessage(content=human))
#         history_langchain_format.append(AIMessage(content=ai))
    
#     # Append the new user message to the history
#     history_langchain_format.append(HumanMessage(content=message))
    
#     # Generate a context-aware query
#     context = "\n".join([msg.content for msg in history_langchain_format])
    
#     # Get the LLM response
#     llm_response = qa_chain({"query": context})
#     answer = llm_response['result']
#     source_docs = llm_response['source_documents']
#     sources = "\n".join([doc.page_content for doc in source_docs])
    
#     # Update the history with the new message and response
#     response_text = f"Answer: {answer}\n\nSources:\n{sources}"
#     history.append((message, response_text))
    
#     return history, history

# # Create the Gradio interface with streaming
# def chat_interface_streaming():
#     with gr.Blocks() as demo:
#         chat = gr.Chatbot()
#         state = gr.State([])
        
#         def respond(message, state):
#             state, new_state = predict(message, state)
#             return state, new_state
        
#         with gr.Row():
#             txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        
#         txt.submit(respond, [txt, state], [chat, state], queue=True)
    
#     demo.launch()

# # Launch the interface
# chat_interface_streaming()

### end block 1


### Block 2 - works fine for self query retreiver in chat interface - no history recall

# def predict(message, history):
#     llm_response = qa_chain({"query": message})
#     answer = llm_response['result']
#     source_docs = llm_response['source_documents']
#     sources = "\n".join([doc.page_content for doc in source_docs])
#     return f"Answer: {answer}\n\nSources:\n{sources}"

# gr.ChatInterface(predict).launch()

### end block 2