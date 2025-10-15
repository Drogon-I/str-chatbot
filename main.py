import streamlit as st
import pandas as pd
from langchain_experimental.agents import  create_pandas_dataframe_agent
from langchain_groq import ChatGroq

st.set_page_config("LLMs con DataFrames")
st.title("LLMs con DataFrames")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= st.secrets["groq"]["API_KEY"],
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def reloadChat():
    st.session_state.messages = []

file = st.file_uploader("Elige un archivo csv", type=["csv"], on_change=reloadChat)
if file is not None:
    df = pd.read_csv(file)
    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
        agent_type="openai-tools", 
        allow_dangerous_code=True)

if prompt := st.chat_input("Pregunta"):
    st.session_state.message = []
    prompt_final = f"Eres un experto de datos, todas tus respuestas son en el idioma español con la siguiente pregunta {prompt} y cuando pida mas de un registro damelo en tablas o en listas de markdown"
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content":prompt_final})

    with st.spinner("Validando datos..."):
        response = agent.run(st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant", "content":response})