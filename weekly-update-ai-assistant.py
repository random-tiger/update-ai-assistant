import uuid
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize memory and agent
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# Step 1: Define the Tavily search tool using @tool decorator
@tool
def search_tavily(query: str) -> str:
    """Search Tavily and return the top results."""
    search = TavilySearchResults(max_results=2)
    return search.run(query)

# Step 2: Define the tool to retrieve CSV, split, embed, and store embeddings
@tool
def search_tubi_launches_embeddings(query: str) -> str:
    """Fetch CSV data, split, embed, and search embeddings."""
    # Step 1: Fetch CSV data
    url = "https://app.periscopedata.com/api/adrise:tubi/chart/csv/9609090c-4c3d-e932-06eb-68353433d860/1460174"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Failed to fetch data from Periscope Data."

    # Step 2: Parse CSV data
    data = StringIO(response.text)
    df = pd.read_csv(data)
    
    # Step 3: Convert each row of the dataframe to a document (e.g., treat each experiment as a document)
    rows = df.apply(lambda row: row.to_string(), axis=1).tolist()
    
    # Step 4: Embed the documents using OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Step 5: Store embeddings in FAISS or another vector database
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents(rows)  # Create documents from rows
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Step 6: Perform a search over the stored embeddings
    search_results = vectorstore.similarity_search(query)
    
    # Step 7: Return the top result (or top N results) formatted as needed
    if search_results:
        # Format the top result, assuming each result is a row representing an experiment
        return f"Relevant data:\n{search_results[0].page_content}"
    else:
        return "No relevant data found in the CSV."

# Combine the tools into the agent's available tools
tools = [search_tavily, search_tubi_launches_embeddings]  # Add both tools to the agent's available tools
agent_executor = create_react_agent(model, tools, checkpointer=memory)  # Memory checkpointer is added back

# App title and description
st.title("Interactive AI Agent with Multiple Tools and Memory")
st.write("Ask the AI anything, and it will retrieve information or answer your question using its available tools.")

# Initialize or retrieve conversation thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # Generate new thread ID
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Initialize conversation history

# User input section
user_question = st.text_input("Please enter your question (e.g., 'what is the weather in SF'):")

if user_question:
    st.write(f"User: {user_question}")

    # Prepare the message with the required 'role' and 'content' keys
    message = {"role": "user", "content": user_question}

    # Include past conversation in the messages if there is any
    past_messages = [{"role": "system", "content": str(msg)} for msg in st.session_state.conversation_history]
    past_messages.append(message)

    # Set the configuration required by the memory checkpointer
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,  # Use the stored thread ID for checkpointing
            "checkpoint_id": str(uuid.uuid4()),  # Generate a unique checkpoint ID for each interaction
        }
    }

    # Execute the agent and stream results
    try:
        for chunk in agent_executor.stream(
            {"messages": past_messages, "thread_id": st.session_state.thread_id}, config
        ):
            # Debug: Print the structure of chunk to identify its keys
            st.write(chunk)  # Display the full chunk to inspect its structure

            # Extract the 'AIMessage' content for the response
            agent_message = chunk["agent"]["messages"][0] if "agent" in chunk and "messages" in chunk["agent"] else "No content found"

            # Store the response in conversation history
            st.session_state.conversation_history.append(user_question)  # Add user question
            st.session_state.conversation_history.append(agent_message)  # Add agent response
            st.write(agent_message)  # Display the agent's response
            st.write("----")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Option to start a new conversation
if st.button("Start New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())  # Generate new thread ID
    st.session_state.conversation_history.clear()  # Clear conversation history
    st.write(f"New thread ID: {st.session_state.thread_id}")

# Display past conversation
st.write("## Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
