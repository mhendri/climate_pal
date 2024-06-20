import openai
import chromadb
import gradio as gr
import os
import json

from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

if not os.environ.get("OPENAI_API_KEY"):
    import key
    key.init()
    assert os.environ.get('OPENAI_API_KEY')

docs = [
    Document(
        page_content="Mole fraction of methane in air calculates the ratio of the number of moles of methane to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of methane, a significant greenhouse gas, in the Earth's atmosphere. Understanding the mole fraction of methane in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole fraction of methane in air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of CH4",
            "out_name": "ch4",
        },
    ),
    Document(
        page_content="cloud area fraction is the total cloud cover percentage and it represents the proportion of the sky covered by clouds when viewed from either the Earth's surface or the top of the atmosphere. This measurement encompasses the entire atmospheric column and includes all types of clouds, both large-scale (such as stratus or cirrus clouds) and convective clouds (like cumulus clouds formed from updrafts). It is a key parameter in meteorology and climate science, as it helps in understanding cloud cover patterns, which play a crucial role in Earth's energy balance, weather forecasting, and climate modeling. The total cloud area fraction is essential for assessing how clouds affect solar radiation, precipitation, and temperature.n climate change and air quality.",
        metadata={
            "standard_name": "cloud area fraction",
            "units": "%",
            "long_name": "Total Cloud Cover Percentage",
            "out_name": "clt",
        },
    ),
    Document(
        page_content="mole fraction of carbon dioxide in air calculates the ratio of the number of moles of carbon dioxide to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of carbon dioxide, a significant greenhouse gas, in the Earth's atmosphere. Understanding the mole fraction of carbon dioxide in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole fraction of carbon dioxide in air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of CO2",
            "out_name": "co2",
        },
    ),
    Document(
        page_content="relative humidity is measured with respect to liquid water for temperatures above 0°C and with respect to ice for temperatures below 0°C",
        metadata={
            "standard_name": "relative humidity",
            "units": "%",
            "long_name": "Relative Humidity",
            "out_name": "hur",
        },
    ),
    Document(
        page_content="relative humidity at a height of 2 meters is measured with respect to liquid water for temperatures above 0°C and with respect to ice for temperatures below 0°C. relative humidity is a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature",
        metadata={
            "standard_name": "relative humidity",
            "units": "%",
            "long_name": "Near-Surface Relative Humidity",
            "out_name": "hurs",
        },
    ),
    Document(
        page_content="specific humidity is an absolute measure of the actual amount of water vapor present in the air. It is not a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature. Specific humidity is an absolute measure of the actual amount of water vapor present in the air.",
        metadata={
            "standard_name": "specific humidity",
            "units": "1",
            "long_name": "Specific Humidity",
            "out_name": "hus",
        },
    ),
    Document(
        page_content="specific humidity near surface at a height of 2 meters is an absolute measure of the actual amount of water vapor present in the air. It is not a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature. Specific humidity is an absolute measure of the actual amount of water vapor present in the air.",
        metadata={
            "standard_name": "specific humidity near surface",
            "units": "1",
            "long_name": "Near-Surface Specific Humidity",
            "out_name": "huss",
        },
    ),
    Document(
        page_content="mole fraction of nitrous oxide in air calculates the ratio of the number of moles of nitrous oxide to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of nitrous oxide in the Earth's atmosphere. Understanding the mole fraction of nitrous oxide in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole fraction of nitrous oxide in air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of N2O",
            "out_name": "n2o",
        },
    ),
    Document(
        page_content="mole fraction of ozone in air calculates the ratio of the number of moles of ozone to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of ozone in the Earth's atmosphere. Understanding the mole fraction of ozone in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole fraction of ozone in air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of O3",
            "out_name": "o3",
        },
    ),
    Document(
        page_content="precipitation flux is the flux of water equivalent (rain or snow) reaching the land surface. This includes the liquid and solid phases of water.",
        metadata={
            "standard_name": "precipitation flux",
            "units": "kg m-2 s-1",
            "long_name": "Precipitation",
            "out_name": "pr",
        },
    ),
    Document(
        page_content="snowfall flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain",
        metadata={
            "standard_name": "snowfall flux",
            "units": "kg m-2 s-1",
            "long_name": "Snowfall Flux",
            "out_name": "prsn",
        },
    ),
    Document(
        page_content="day snowfall flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain",
        metadata={
            "standard_name": "snowfall flux",
            "units": "kg m-2 s-1",
            "long_name": "Snowfall Flux",
            "out_name": "prsn",
        },
    ),
    Document(
        page_content="surface pressure is the pressure (force per unit area) of the air at the lower boundary of the atmosphere. It is a measure of the weight that all the air in a column vertically above a point on the Earth's surface. It is calculated over all surfaces - land, sea and inland water.",
        metadata={
            "standard_name": "surface air pressure",
            "units": "Pa",
            "long_name": "Surface Air Pressure",
            "out_name": "ps",
        },
    ),
    Document(
        page_content="air pressure at mean sea level is the pressure (force per unit area) of the atmosphere at the surface of the Earth, adjusted to the height of sea level. It is a measure of the weight that all the air in a column vertically above a point on the Earth's surface would have, if the point were located at sea level. It is calculated over all surfaces - land, sea and inland water.",
        metadata={
            "standard_name": "air pressure at mean sea level",
            "units": "Pa",
            "long_name": "Sea Level Pressure",
            "out_name": "psl",
        },
    ),
    Document(
        page_content="wind speed is the magnitude of the two-dimensional horizontal air velocity near the surface.",
        metadata={
            "standard_name": "wind speed",
            "units": "m s-1",
            "long_name": "Near-Surface Wind Speed",
            "out_name": "sfcWind",
        },
    ),
    Document(
        page_content="air temperature is the temperature in the atmosphere. It has units of Kelvin (K). Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15. This parameter is available on multiple levels through the atmosphere.",
        metadata={
            "standard_name": "air temperature",
            "units": "K",
            "long_name": "Air Temperature",
            "out_name": "ta",
        },
    ),
    Document(
        page_content="air temperature near surface is the temperature of air at 2 meters above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air temperature near surface",
            "units": "K",
            "long_name": "Near-Surface Air Temperature",
            "out_name": "tas",
        },
    ),
    
    Document(
        page_content="air temperature daily maximum is the daily maximum temperature of air at 2m above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air temperature daily maximum",
            "units": "K",
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "out_name": "tasmax",
        },
    ),
    Document(
        page_content="air temperature daily minimum is the daily minimum temperature of air at 2m above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air temperature daily minimum",
            "units": "K",
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "out_name": "tasmin",
        },
    ),
    Document(
        page_content="eastward wind is the magnitude of the eastward component of the two-dimensional horizontal air velocity 10 meters above the surface",
        metadata={
            "standard_name": "eastward wind",
            "units": "m s-1",
            "long_name": "Eastward Near-Surface Wind",
            "out_name": "uas",
        },
    ),
    Document(
        page_content="northward wind is the magnitude of the northward component of the two-dimensional horizontal air velocity 10 meters above the surface",
        metadata={
            "standard_name": "northward wind",
            "units": "m s-1",
            "long_name": "Northward Near-Surface Wind",
            "out_name": "vas",
        },
    ),
]

vectorstore = Chroma.from_documents(
    docs,
    OpenAIEmbeddings(),
)


metadata_field_info = [
    AttributeInfo(
        name="frequency",
        description="The frequency of the data",
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
        name="out_name",
        description="the variable name in the dataset",
        type="string",
    ),
]

document_content_description = "climate diagnostic variables"
llm = ChatOpenAI(
    temperature=0, 
)
# based on conversational RAG tutorial https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history
retriever = vectorstore.as_retriever()
system_prompt = (
    "You are a NASA climate scientist. Answer your colleague's questions about the CMIP6 dataset, asking for clarification if needed. "
    "Also, provide a set of exactly three suggestions of what your colleague might say next. "
    "Provide your responses in the following format, with text prepended by 'Answer: ' and followed by a list of reply suggestions and "
    "a list of your sources like in the following example:\n"
    "\"Answer: Rain is liquid water droplets that fall from clouds towards the Earth's surface. It is a form of precipitation that "
    "occurs when water vapor in the atmosphere condenses into water droplets that become heavy enough to fall due to gravity.\n\n"
    "Sources:\n"
    "- precipitation flux is the flux of water equivalent (rain or snow) reaching the land surface. This includes the liquid and solid phases of water.\n"
    "- snowfall flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain\n"
    "- day snowfall flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain\n"
    "- relative humidity at a height of 2 meters is measured with respect to liquid water for temperatures above 0°C and with respect to ice for temperatures below 0°C. relative humidity is a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature\n\n"
    "Suggestions:\n"
    "- What variables have a large impact on rain?\n"
    "- Plot average precipitation flux over the Earth for the last 5 years.\n"
    "- Describe trends in precipitation flux over the last 50 years in New York City.\"\n\n"
    "Use the following sources to formulate your responses:\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)
doc_chain = create_stuff_documents_chain(llm, prompt) #gives prompt to LLM
qa_chain = create_retrieval_chain(retriever, doc_chain) #gives documents to prompt

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    "Be as concise as possible, using only a few key words."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# this processes chat history on call into standalone query:
history_aware_retriever = create_history_aware_retriever( 
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# connects standalone prompt to answer with docs etc
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
# connects history->standalone with standalone->answer
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) 

def predict(message, historystr, historychn):

    ai_message = rag_chain.invoke({'input': message, 'chat_history': historychn})
    print(ai_message)

    # source_docs = ai_message['context']
    # sources = "\n".join([doc.page_content for doc in source_docs])
    # response_text = f"Answer: {ai_message['answer']}\n\nSources:\n{sources}"
    response_text = str(ai_message['answer'])
    historystr.append(
        (message, response_text)
    )
    historychn.extend(
        [
            HumanMessage(content=message),
            AIMessage(content=ai_message['answer'])
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