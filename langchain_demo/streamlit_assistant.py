import os
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pprint import pprint
import time
import requests
import os
import streamlit as st


def create_file( client, paths=["pr_Amon_GISS-E2-1-G_ssp245_r10i1p1f2_gn_201501-205012.nc"]):
    files = []
    for path in paths:
        file = client.files.create(
            file=open(path, "rb"),
            purpose='assistants'
        )
        files.append(file)
    
    return files
        
    # file = client.files.create(
    #     file=open(path, "rb"),
    #     purpose='assistants'
    # )
    # return file


def create_assistant(files, client):
    assistant = client.beta.assistants.create(
        name="Climate PAL Assistant",
        instructions="You are a climate scientist that is an expert in analyzing and plotting data. When asked a climate science question, write and run code to answer the question. Make sure to only talk about the question and do not mention anything else such as difficulty decoding and missing modules in you python environment. Use the xarray library to read .nc files.", #Use the cftime library to decode time variables",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
        # I need to figure out what this does
        tool_resources={
        "code_interpreter": {
        "file_ids": [file.id for file in files] # modified this line
        }
        }
    )
    return assistant
  
def create_thread(client):
  thread = client.beta.threads.create()
  return thread
  
def create_message(thread, client, text_body=''):
  message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content = text_body
  )
  print("Here is the message we just created: ", message)
  return message


def download_file(url):
    # Send a HTTP request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract the filename from the URL
        local_filename = os.path.basename(url)
        
        # Open a local file with write-binary mode
        with open("data/"+local_filename, "wb") as file:
            # Write the contents of the response to the file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download completed successfully. File saved as {local_filename}")
    else:
        print("Failed to download the file. Status code:", response.status_code)
        
    return local_filename    
  
    
def print_user(text):
    avatar_url = "https://avataaars.io/?avatarStyle=Transparent&topType=LongHairDreads&accessoriesType=Wayfarers&hairColor=BrownDark&facialHairType=Blank&clotheType=CollarSweater&clotheColor=Blue03&eyeType=Close&eyebrowType=Default&mouthType=Twinkle&skinColor=Brown"
    message_alignment = "flex-end"
    message_bg_color = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
    avatar_class = "user-avatar"
    #            <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
    
    st.write(
        f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                    {text} \n </div>
            </div>
            """,
        unsafe_allow_html=True,
        )


def print_bot(text, image_path=None):
    avatar_url = 'https://avataaars.io/?avatarStyle=Transparent&topType=WinterHat1&accessoriesType=Blank&hatColor=Red&facialHairType=Blank&clotheType=Hoodie&clotheColor=Gray01&eyeType=Happy&eyebrowType=Default&mouthType=Smile&skinColor=Light'
    message_alignment = "flex-start"
    message_bg_color = "#EEEEEE"
    avatar_class = "bot-avatar"

    if image_path:
        st.image(image_path, use_column_width=True)
         #           <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
        
        
    else:
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                    <div style="background: {message_bg_color}; color: black; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                        {text} \n </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
    
    
    
    
def main():
    st.set_page_config(page_title="Climate Pal", initial_sidebar_state='auto', menu_items=None)
    st.title("Climate Pal 🌎🌪️❄️🌊")
    st.caption('Navigate the CMIP data with ease!!!')
    
    # set key for openai
    if not os.environ.get("OPENAI_API_KEY"):
        import key
        key.init()
        assert os.environ.get('OPENAI_API_KEY')
        
    with st.spinner('Setting up assistant...'):
        # create client
        if "client" not in st.session_state.keys():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            st.session_state["client"] = client
            
        # create file
        if "file" not in st.session_state.keys():
            file = create_file(st.session_state["client"], paths=["data/hus_Amon_GISS-E2-1-G_historical_r1i1p1f2_gn_200101-201412.nc","pr_Amon_GISS-E2-1-G_ssp245_r10i1p1f2_gn_201501-205012.nc"])
            st.session_state["file"] = file
        
        # create assistant
        if "assistant" not in st.session_state.keys():
            assistant = create_assistant(st.session_state["file"], st.session_state["client"])
            st.session_state["assistant"] = assistant
            
        # create thread
        if "thread" not in st.session_state.keys():
            thread = create_thread(st.session_state["client"])
            st.session_state["thread"] = thread
        

    
    print_bot("Hey there, I'm Climate Pal, your personal data visualizer and analyzer, ready to explore some data?! Please tell me what you would like to do.")

    if input := st.chat_input(placeholder="Type here..."):
        
        with st.spinner('In Progress...'):
            # create message
            message = create_message(st.session_state["thread"], st.session_state["client"], input)
            
            run = st.session_state["client"].beta.threads.runs.create(
                thread_id=st.session_state["thread"].id,
                assistant_id=st.session_state["assistant"].id,
            )
            
            run = st.session_state['client'].beta.threads.runs.retrieve(thread_id=st.session_state['thread'].id, run_id=run.id)
            print(run.status)
            
            while run.status not in ["completed", "failed"]:
                run = st.session_state['client'].beta.threads.runs.retrieve(
                    thread_id = st.session_state['thread'].id,
                    run_id = run.id
                )

                print(run.status)
                time.sleep(5)
            
            messages = st.session_state['client'].beta.threads.messages.list(thread_id=st.session_state["thread"].id)

        
        for message in reversed(messages.data):
            content_block = message.content[0]
            
            if hasattr(content_block, 'text'):
                if message.role == "assistant":
                    print_bot(content_block.text.value)
                else:
                    print_user(content_block.text.value)
                
                print(message.role + ": " + content_block.text.value)
                
            elif hasattr(content_block, 'image_file'):
                
                api_response = st.session_state['client'].files.with_raw_response.retrieve_content(content_block.image_file.file_id)

                content = api_response.content
                with open(f"assistant_images/{content_block.image_file.file_id}.png", 'wb') as f:
                    f.write(content)
                
                image_path = f"assistant_images/{content_block.image_file.file_id}.png"
                
                print_bot("[Non-text content]", image_path)





if __name__ == "__main__":
    main()