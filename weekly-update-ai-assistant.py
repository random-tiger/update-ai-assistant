import uuid
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize search tool
search = TavilySearchResults(max_results=2)

# Initialize memory and agent
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# App title and description
st.title("Interactive AI Agent")
st.write("Ask the AI anything, and it will use various tools to retrieve and generate responses.")

# Select conversation thread management
config = {"configurable": {"thread_id": "abc123"}}
conversation_history = []

# Start new conversation
thread_id = str(uuid.uuid4())
st.write(f"New thread ID: {thread_id}")

# User input section
user_question = st.text_input("Please enter your question:")

if user_question:
    st.write(f"User: {user_question}")
    
    # Prepare the message with the required 'role' and 'content' keys
    message = {"role": "user", "content": user_question}

    # Execute the agent and stream results
    try:
        for chunk in agent_executor.stream(
            {"messages": [message], "thread_id": thread_id}, config
        ):
            conversation_history.append(chunk)
            st.write(chunk)
            st.write("----")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Option to start a new conversation
if st.button("Start New Conversation"):
    thread_id = str(uuid.uuid4())
    st.write(f"New thread ID: {thread_id}")
    conversation_history.clear()

# Display past conversation
st.write("## Conversation History")
for message in conversation_history:
    st.write(message)
