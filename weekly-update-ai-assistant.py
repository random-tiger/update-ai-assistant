import uuid
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode  # Import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import START, StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize memory and agent with memory saving and tracing enabled
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, tracing_v2=langchain_tracing_v2)

# Step 1: Define the Tavily search tool
@tool
def search_tavily(query: str) -> str:
    """Search Tavily and return the top results."""
    search = TavilySearchResults(max_results=2)
    return search.run(query)

# Step 2: Define the tool to retrieve CSV, split, embed, and store embeddings
@tool
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

# Step 3: Handle tool error fallback
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Step 4: Create tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Step 5: Define the main assistant
class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state, config):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (not result.content):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Step 6: Combine tools into the assistant
tools = [search_tavily, search_csv_embeddings]
llm_assistant = model.bind_tools(tools)

# Step 7: Define the state schema for the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Step 8: Build the StateGraph for handling assistant interaction with memory and state
state_graph = StateGraph(State)

# Define the assistant node and tool node
state_graph.add_node("assistant", Assistant(llm_assistant))
state_graph.add_node("tools", create_tool_node_with_fallback(tools))

# Define the flow from assistant to tools and back
state_graph.add_edge(START, "assistant")
state_graph.add_conditional_edges("assistant", tools_condition)
state_graph.add_edge("tools", "assistant")

# Compile the graph with memory saving and tracing enabled
agent_graph = state_graph.compile(checkpointer=memory, tracing_v2=langchain_tracing_v2)

# Streamlit UI Setup
st.title("Interactive AI Agent with State and Memory")
st.write("Ask the AI anything, and it will retrieve information or answer your question using its available tools.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

user_question = st.text_input("Please enter your question:")

if user_question:
    st.write(f"User: {user_question}")
    
    state = {"messages": st.session_state.conversation_history}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        result = agent_graph.invoke(state, config)
        agent_message = result["messages"][-1]["content"]
        st.session_state.conversation_history.append({"role": "assistant", "content": agent_message})
        st.write(agent_message)
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Start New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.conversation_history.clear()
    st.write(f"New conversation started with ID: {st.session_state.thread_id}")
