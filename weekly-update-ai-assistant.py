import uuid
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from datetime import datetime

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
tavily_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize memory and model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# Define the Tavily search tool
@tool
def search_tavily(query: str) -> str:
    """Search Tavily and return the top results."""
    search = TavilySearchResults(max_results=2)
    return search.run(query)

# Define the tool to retrieve CSV, split, embed, and search embeddings
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

# Prompt template to structure the conversation
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the provided tools to search for information. "
            "If no results are found, persistently expand the query bounds."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Combine tools into a runnable assistant
tools = [search_tavily, search_csv_embeddings]
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)

# Assistant class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            state = {**state, "user_info": configuration.get("user_info", None)}
            result = self.runnable.invoke(state)
            
            # Check if result is empty or needs re-prompting
            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Streamlit UI Setup
st.title("Interactive AI Assistant")
st.write("Ask the AI anything, and it will use tools or provide answers.")

# Initialize session state for thread and conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_question = st.text_input("Please enter your question:")

if user_question:
    st.write(f"User: {user_question}")
    
    # Update conversation history with user's question
    st.session_state.conversation_history.append({"role": "user", "content": user_question})
    
    # Prepare the assistant state with messages
    state = {"messages": st.session_state.conversation_history}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Create an assistant instance and invoke it
    assistant = Assistant(assistant_runnable)
    try:
        result = assistant(state, config)
        assistant_message = result["messages"][-1]["content"]
        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_message})
        st.write(assistant_message)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Option to start a new conversation
if st.button("Start New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.conversation_history.clear()
    st.write(f"New conversation started with ID: {st.session_state.thread_id}")
