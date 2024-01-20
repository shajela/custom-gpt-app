import os
import streamlit as st
import openai

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_API_KEY")

with st.sidebar:
    st.title("About")
    st.markdown('''
        This app was built using
        - [Streamlit](https://streamlit.io/)
        - [Llama Index](https://gpt-index.readthedocs.io/)
        - [OpenAI](https://platform.openai.com/docs/overview)

        You can ask it questions about common programming
        languages and methods such as Python, C++, SQL, etc.
    ''')

st.header("Learn something new.")

reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
docs = reader.load_data()
service_context = ServiceContext.from_defaults(llm=OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
))
index = VectorStoreIndex.from_documents(docs, service_context=service_context)

query = st.text_input("Ask test...")
if query:
    chat_engine = index.as_chat_engine(chat_mode="condense_question")
    response = chat_engine.chat(query)
    st.write(response.response)