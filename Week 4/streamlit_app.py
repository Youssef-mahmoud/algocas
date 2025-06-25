# streamlit_app.py

import os
import getpass
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Load environment variables ---
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY") or getpass.getpass("Enter GROQ API Key: ")
tavily_key = os.getenv("TAVILY_API_KEY") or getpass.getpass("Enter Tavily API Key: ")
os.environ["TAVILY_API_KEY"] = tavily_key

# --- LLM + Tools ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_key,
    model="llama3-70b-8192"
)
search_tool = TavilySearchResults(k=2)

# --- Shared Memory & Agent ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        memory=st.session_state.memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )

# --- Streamlit Chat UI ---
st.set_page_config(page_title="🦙 LLaMA 3 Agent", page_icon="🦙")
st.title("🦙 LLaMA 3 Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if user_input:
    response = st.session_state.agent.run(user_input)
    st.session_state.chat_history.extend([
        ("You", user_input),
        ("Agent", response)
    ])

# --- Display chat history
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

# --- Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        memory=st.session_state.memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )
    st.rerun()
