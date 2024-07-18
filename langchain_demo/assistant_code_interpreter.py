from openai import OpenAI

client = OpenAI(
  api_key=''
)
  
# Upload a file with an "assistants" purpose
file = client.files.create(
  file=open("pr_Amon_GISS-E2-1-G_ssp245_r10i1p1f2_gn_201501-205012.nc", "rb"),
  purpose='assistants'
)

# Create an assistant using the file ID
assistant = client.beta.assistants.create(
  name="Climate PAL Assistant",
  instructions="You are a climate scientist. When asked a climate science question, write and run code to answer the question. Use the xarray library to read .nc files. Download and use the cftime library to decode time variables",
  model="gpt-4o",
  tools=[{"type": "code_interpreter"}]
)

thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "Create a plot of total rainfall",
      "attachments": [
        {
          "file_id": file.id,
          "tools": [{"type": "code_interpreter"}]
        }
      ]
    }
  ]
)



from typing_extensions import override
from openai import AssistantEventHandler
 
# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.
 
class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)
 
# Then, we use the `stream` SDK helper 
# with the `EventHandler` class to create the Run 
# and stream the response.
 
with client.beta.threads.runs.stream(
  thread_id=thread.id,
  assistant_id=assistant.id,
  # instructions="Please address the user as Jane Doe. The user has a premium account.",
  event_handler=EventHandler(),
) as stream:
  stream.until_done()

  
  # how do we maintain a back-and-forth thread with the agent?