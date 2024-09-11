import uuid
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool

# Access secrets for API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Initialize memory and agent
memory = MemorySaver()
model = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0,
    max_tokens=500
)

# Define the Tavily search tool
@tool
def search_tavily(query: str) -> str:
    """Search Tavily and return the top results."""
    search = TavilySearchResults(max_results=2)
    return search.run(query)

# Define the tool to retrieve CSV, split, embed, and store embeddings
@tool
def search_tubi_launches_embeddings(query: str) -> str:
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

# Define the tool to reformat responses based on the style guide
@tool
def format_agent_response_llm(response: str) -> str:
    """Use GPT-4o to reformat the agent's response based on a style guide."""
    
    # Step 1: Fetch the style guide from GitHub or a local file
    style_guide_url = "https://github.com/random-tiger/update-ai-assistant/raw/3f69cfd7f15542d1783f1c083259a86e5bf43016/style-guide.md"
    style_guide_response = requests.get(style_guide_url)
    
    if style_guide_response.status_code != 200:
        return "Failed to retrieve the style guide."
    
    style_guide = style_guide_response.text
    
    # Step 2: Use GPT-4o to apply the style guide to the agent's response
    messages = [
        {"role": "system", "content": f"Below is a style guide:\n\n{style_guide}"},
        {"role": "user", "content": f"Here is the agent's response:\n\n{response}\n\nReformat it to adhere to the style guide."}
    ]
    
    # Ensure the correct method is used for GPT-4o
    try:
        # Send the prompt to the GPT-4o model and extract the response
        llm_response = model.invoke(messages)  # Use invoke for GPT-4o
        
        # Extract the reformatted response from the model's output
        reformatted_response = llm_response.content.strip()
    
    except KeyError:
        return "Failed to generate reformatted response due to missing fields in GPT-4o output."
    except Exception as e:
        return f"An error occurred while generating the response: {e}"
    
    return reformatted_response

# Combine the tools into the agent's available tools
tools = [search_tavily, search_tubi_launches_embeddings, format_agent_response_llm]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# App title and description
st.title("Interactive AI Agent with Multiple Tools and Memory")
st.write("Ask the AI anything, and it will retrieve information or answer your question using its available tools.")

# Initialize or retrieve conversation thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input section
user_question = st.text_input("Please enter your question (e.g., 'what is the weather in SF'):")

if user_question:
    st.write(f"User: {user_question}")

    message = {"role": "user", "content": user_question}
    past_messages = [{"role": "system", "content": str(msg)} for msg in st.session_state.conversation_history]
    past_messages.append(message)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "checkpoint_id": str(uuid.uuid4()),
        }
    }

    try:
        for chunk in agent_executor.stream(
            {"messages": past_messages, "thread_id": st.session_state.thread_id}, config
        ):
            st.write(chunk)
            agent_message = chunk["agent"]["messages"][0] if "agent" in chunk and "messages" in chunk["agent"] else "No content found"
            
            # Reformat the response using the LLM-based tool
            formatted_message = format_agent_response_llm(agent_message)
            
            st.session_state.conversation_history.append(user_question)
            st.session_state.conversation_history.append(formatted_message)
            st.write(formatted_message)
            st.write("----")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Start New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.conversation_history.clear()
    st.write(f"New thread ID: {st.session_state.thread_id}")

st.write("## Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)
