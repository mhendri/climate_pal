import openai
import chromadb
import gradio as gr
import os
import json

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


docs = [
    Document(
        page_content="Mole_fraction_of_methane_in_air calculates the ratio of the number of moles of methane to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of methane, a significant greenhouse gas, in the Earth's atmosphere. Understanding the mole fraction of methane in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole_fraction_of_methane_in_air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of CH4",
            "out_name": "ch4",
        },
    ),
    Document(
        page_content="cloud_area_fraction is the total cloud cover percentage and it represents the proportion of the sky covered by clouds when viewed from either the Earth's surface or the top of the atmosphere. This measurement encompasses the entire atmospheric column and includes all types of clouds, both large-scale (such as stratus or cirrus clouds) and convective clouds (like cumulus clouds formed from updrafts). It is a key parameter in meteorology and climate science, as it helps in understanding cloud cover patterns, which play a crucial role in Earth's energy balance, weather forecasting, and climate modeling. The total cloud area fraction is essential for assessing how clouds affect solar radiation, precipitation, and temperature.n climate change and air quality.",
        metadata={
            "standard_name": "cloud_area_fraction",
            "units": "%",
            "long_name": "Total Cloud Cover Percentage",
            "out_name": "clt",
        },
    ),
    Document(
        page_content="mole_fraction_of_carbon_dioxide_in_air calculates the ratio of the number of moles of carbon dioxide to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of carbon dioxide, a significant greenhouse gas, in the Earth's atmosphere. Understanding the mole fraction of carbon dioxide in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole_fraction_of_carbon_dioxide_in_air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of CO2",
            "out_name": "co2",
        },
    ),
    Document(
        page_content="relative_humidity is measured with respect to liquid water for temperatures above 0°C and with respect to ice for temperatures below 0°C",
        metadata={
            "standard_name": "relative_humidity",
            "units": "%",
            "long_name": "Relative Humidity",
            "out_name": "hur",
        },
    ),
    Document(
        page_content="relative_humidity at a height of 2 meters is measured with respect to liquid water for temperatures above 0°C and with respect to ice for temperatures below 0°C. relative_humidity is a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature",
        metadata={
            "standard_name": "relative_humidity",
            "units": "%",
            "long_name": "Near-Surface Relative Humidity",
            "out_name": "hurs",
        },
    ),
    Document(
        page_content="specific_humidity is an absolute measure of the actual amount of water vapor present in the air. It is not a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature. Specific humidity is an absolute measure of the actual amount of water vapor present in the air.",
        metadata={
            "standard_name": "specific_humidity",
            "units": "1",
            "long_name": "Specific Humidity",
            "out_name": "hus",
        },
    ),
    Document(
        page_content="specific_humidity_near_surface at a height of 2 meters is an absolute measure of the actual amount of water vapor present in the air. It is not a ratio of the amount of water vapor present to the maximum amount of water vapor the air can hold at a given temperature. Specific humidity is an absolute measure of the actual amount of water vapor present in the air.",
        metadata={
            "standard_name": "specific_humidity_near_surface",
            "units": "1",
            "long_name": "Near-Surface Specific Humidity",
            "out_name": "huss",
        },
    ),
    Document(
        page_content="mole_fraction_of_nitrous_oxide_in_air calculates the ratio of the number of moles of nitrous oxide to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of nitrous oxide in the Earth's atmosphere. Understanding the mole fraction of nitrous oxide in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole_fraction_of_nitrous_oxide_in_air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of N2O",
            "out_name": "n2o",
        },
    ),
    Document(
        page_content="mole_fraction_of_ozone_in_air calculates the ratio of the number of moles of ozone to the total number of moles of all gases present in the air. This measurement is particularly important in environmental science and atmospheric studies, as it provides a precise way to determine the concentration of ozone in the Earth's atmosphere. Understanding the mole fraction of ozone in air is crucial for assessing its impact on climate change and air quality.",
        metadata={
            "standard_name": "mole_fraction_of_ozone_in_air",
            "units": "mol mol-1",
            "long_name": "Mole Fraction of O3",
            "out_name": "o3",
        },
    ),
    Document(
        page_content="precipitation_flux is the flux of water equivalent (rain or snow) reaching the land surface. This includes the liquid and solid phases of water.",
        metadata={
            "standard_name": "precipitation_flux",
            "units": "kg m-2 s-1",
            "long_name": "Precipitation",
            "out_name": "pr",
        },
    ),
    Document(
        page_content="snowfall_flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain",
        metadata={
            "standard_name": "snowfall_flux",
            "units": "kg m-2 s-1",
            "long_name": "Snowfall Flux",
            "out_name": "prsn",
        },
    ),
    Document(
        page_content="day snowfall_flux is the flux of water equivalent in solid form only that reaches the land surface. It does not include liqiud form also known as rain",
        metadata={
            "standard_name": "snowfall_flux",
            "units": "kg m-2 s-1",
            "long_name": "Snowfall Flux",
            "out_name": "prsn",
        },
    ),
    Document(
        page_content="surface_pressure is the pressure (force per unit area) of the air at the lower boundary of the atmosphere. It is a measure of the weight that all the air in a column vertically above a point on the Earth's surface. It is calculated over all surfaces - land, sea and inland water.",
        metadata={
            "standard_name": "surface_air_pressure",
            "units": "Pa",
            "long_name": "Surface Air Pressure",
            "out_name": "ps",
        },
    ),
    Document(
        page_content="air_pressure_at_mean_sea_level is the pressure (force per unit area) of the atmosphere at the surface of the Earth, adjusted to the height of sea level. It is a measure of the weight that all the air in a column vertically above a point on the Earth's surface would have, if the point were located at sea level. It is calculated over all surfaces - land, sea and inland water.",
        metadata={
            "standard_name": "air_pressure_at_mean_sea_level",
            "units": "Pa",
            "long_name": "Sea Level Pressure",
            "out_name": "psl",
        },
    ),
    Document(
        page_content="wind_speed is the magnitude of the two-dimensional horizontal air velocity near the surface.",
        metadata={
            "standard_name": "wind_speed",
            "units": "m s-1",
            "long_name": "Near-Surface Wind Speed",
            "out_name": "sfcWind",
        },
    ),
    Document(
        page_content="air_temperature is the temperature in the atmosphere. It has units of Kelvin (K). Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15. This parameter is available on multiple levels through the atmosphere.",
        metadata={
            "standard_name": "air_temperature",
            "units": "K",
            "long_name": "Air Temperature",
            "out_name": "ta",
        },
    ),
    Document(
        page_content="air_temperature_near_surface is the temperature of air at 2 meters above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air_temperature_near_surface",
            "units": "K",
            "long_name": "Near-Surface Air Temperature",
            "out_name": "tas",
        },
    ),
    
    Document(
        page_content="air_temperature_daily_maximum is the daily maximum temperature of air at 2m above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air_temperature_daily_maximum",
            "units": "K",
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "out_name": "tasmax",
        },
    ),
    Document(
        page_content="air_temperature_daily_minimum is the daily minimum temperature of air at 2m above the surface of land, sea or inland waters.",
        metadata={
            "standard_name": "air_temperature_daily_minimum",
            "units": "K",
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "out_name": "tasmin",
        },
    ),
    Document(
        page_content="eastward_wind is the magnitude of the eastward component of the two-dimensional horizontal air velocity 10 meters above the surface",
        metadata={
            "standard_name": "eastward_wind",
            "units": "m s-1",
            "long_name": "Eastward Near-Surface Wind",
            "out_name": "uas",
        },
    ),
    Document(
        page_content="northward_wind is the magnitude of the northward component of the two-dimensional horizontal air velocity 10 meters above the surface",
        metadata={
            "standard_name": "northward_wind",
            "units": "m s-1",
            "long_name": "Northward Near-Surface Wind",
            "out_name": "vas",
        },
    ),
]

vectorstore = Chroma.from_documents(
    docs,
    OpenAIEmbeddings(
        openai_api_key="" # add an OpenAI API Key
    ),
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
    temperature=0, openai_api_key="" # Add an OpenAI API Key
)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=self_query_retriever,
    return_source_documents=True,
)


def predict(message, history):
    # Convert the history to Langchain format
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    
    # Append the new user message to the history
    history_langchain_format.append(HumanMessage(content=message))
    
    # Generate a context-aware query
    context = "\n".join([msg.content for msg in history_langchain_format])
    
    # Get the LLM response
    llm_response = qa_chain({"query": context})
    answer = llm_response['result']
    source_docs = llm_response['source_documents']
    sources = "\n".join([doc.page_content for doc in source_docs])
    
    # Update the history with the new message and response
    response_text = f"Answer: {answer}\n\nSources:\n{sources}"
    history.append((message, response_text))
    
    return history, history

# Create the Gradio interface with streaming
def chat_interface_streaming():
    with gr.Blocks() as demo:
        chat = gr.Chatbot()
        state = gr.State([])
        
        def respond(message, state):
            state, new_state = predict(message, state)
            return state, new_state
        
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        
        txt.submit(respond, [txt, state], [chat, state], queue=True)
    
    demo.launch(share=True)

# Launch the interface
# this breaks on NASA machines
chat_interface_streaming()

# Alternatively, invoke retriever with one off query
# print(retriever.invoke("What's one variable about snow with frequency mon"))
print(self_query_retriever.invoke("I am interested in air temperature"))