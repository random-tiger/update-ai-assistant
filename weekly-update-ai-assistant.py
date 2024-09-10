import uuid
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize the model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# Define the State as a typed dictionary
class State(TypedDict):
    messages: list

# Define Tools
# Tavily Search Tool
def search_tavily(query: str) -> str:
    """Search Tavily and return the top results."""
    search = TavilySearchResults(max_results=2)
    return search.run(query)

# CSV Embedding Search Tool
def search_csv_embeddings(query: str) -> str:
    """Fetch CSV data, split, embed, and search embeddings."""
    url = "https://app.periscopedata.com/api/adrise:tubi/chart/csv/9609090c-4c3d-e932-06eb-68353433d860/1460174"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Failed to fetch data from Periscope Data."

    data = StringIO(response.text)
    df = pd.read_csv(data)

    rows = df.apply(lambda row: row.to_string(), axis=1).tolist()
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents(rows)
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    search_results = vectorstore.similarity_search(query)
    
    if search_results:
        return f"Relevant data:\n{search_results[0].page_content}"
    else:
        return "No relevant data found in the CSV."

# Assistant Class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            
            # Re-prompt if no result is found
            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                state["messages"].append(HumanMessage(content="Respond with a real output."))
            else:
                break
        return result

# Define the Primary Prompt Template (without user_info)
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the provided tools to search for information."
            "\n\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Define the tools
tools = [search_tavily, search_csv_embeddings]

# Assistant runnable
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)

# Define the graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))

# Define edges for the assistant
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)

# Memory saver for checkpointing
memory = MemorySaver()
state_graph = builder.compile(checkpointer=memory)

# Streamlit UI
st.title("Interactive AI Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

user_question = st.text_input("Please enter your question:")

if user_question:
    st.write(f"User: {user_question}")
    st.session_state.conversation_history.append(HumanMessage(content=user_question))

    # Prepare the state for the assistant
    state = {"messages": st.session_state.conversation_history}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    events = state_graph.stream({"messages": ("user", user_question)}, config, stream_mode="values")

    for event in events:
        # Loop through messages from the assistant response
        for message in event.get("messages", []):
            if isinstance(message, AIMessage):
                st.write(message.content)  # Write the AI's message to the Streamlit UI
                st.session_state.conversation_history.append(AIMessage(content=message.content))  # Append it to conversation history

# Option to start a new conversation
if st.button("Start New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.conversation_history.clear()
    st.write(f"New conversation started with ID: {st.session_state.thread_id}")
