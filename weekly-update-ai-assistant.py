import uuid
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Access secrets for API keys (assuming you set them in secrets.toml)
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize Tavily search tool
search = TavilySearchResults(max_results=2)

# Initialize memory and agent
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
tools = [search]  # Add Tavily search tool to the agent's available tools
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# App title and description
st.title("Interactive AI Agent with Tavily Search")
st.write("Ask the AI anything, and it will retrieve information or answer your question using its available tools.")

# Conversation thread management
config = {"configurable": {"thread_id": "abc123"}}
conversation_history = []

# Start a new conversation
thread_id = str(uuid.uuid4())
st.write(f"New thread ID: {thread_id}")

# User input section
user_question = st.text_input("Please enter your question (e.g., 'what is the weather in SF'):")

if user_question:
    st.write(f"User: {user_question}")
    
    # Prepare the message with the required 'role' and 'content' keys
    message = {"role": "user", "content": user_question}

    final_answer = None  # To store the final agent response

    # Execute the agent and stream results
    try:
        for chunk in agent_executor.stream(
            {"messages": [message], "thread_id": thread_id}, config
        ):
            # Log intermediate steps (for backend logging)
            print("Logging intermediate step: ", chunk)

            # Capture the final AI response to display to the user
            if 'AIMessage' in str(chunk):
                final_answer = chunk  # Store the final AI message

            # Update memory with the agent's output to retain context for future queries
            memory.add(chunk)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Display only the final response to the user
    if final_answer:
        st.write("Final Response:")
        st.write(final_answer)
    else:
        st.write("No final answer generated.")
        
# Option to start a new conversation
if st.button("Start New Conversation"):
    thread_id = str(uuid.uuid4())
    st.write(f"New thread ID: {thread_id}")
    conversation_history.clear()

# Display past conversation (only final responses)
st.write("## Conversation History")
for message in conversation_history:
    st.write(message)
