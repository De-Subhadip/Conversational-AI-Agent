import os
import warnings
import streamlit as st
from uuid import uuid4
from additionals import *
from duckduckgo_search import DDGS
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

warnings.filterwarnings("ignore")

system_message, router_message, summarise_message = prompts() 

### Setting up the LLM: ###
if not os.environ.get("TOGETHER_API_KEY"):
  os.environ["TOGETHER_API_KEY"] = st.secrets['TOGETHER_API_KEY']

LLM = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")
embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

vectorStore = InMemoryVectorStore(embedding_model)

### Setting up the sesssion state, app title and layout: ###
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.mainLLMInput = system_message
    st.session_state.routerLLMInput = router_message

st.set_page_config(layout='wide')

st.markdown("<h2 style='text-align: center;'>What\'s on your mind?</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stVerticalBlock"] h2 {
        margin-top: -40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.html(
    """
<style>
    .stChatInput div {
        min-height: 120px,
    }
</style>
    """
)

with st.sidebar:
    st.title(':red[Llama 4 Scout ]:robot_face:')
    st.divider()
    clear_chat_history = st.button('**Erase Chatbot Memory**', type='primary')
    if clear_chat_history:
        st.session_state.messages = []
        st.session_state.mainLLMInput = system_message


### Setting up the Chatbot UI: ###
_, chatbot_UI, _ = st.columns([1, 3, 1], border=False)

def stream_response():
    from time import sleep
    for token in finalOutput.split(" "):
        yield token + " "
        sleep(0.1)

with chatbot_UI:
    output_container = st.container(height=480, border=False)

    if userInput := st.chat_input("Chat with Llama 4", accept_file=True, file_type=['pdf', 'txt']):
        st.session_state.routerLLMInput += f"User:\n{userInput.text}\n\n"
        routerResponse = LLMResponse(st.session_state.routerLLMInput, LLM)

        if not (routerResponse.startswith('{') and routerResponse.endswith('}')):
            st.session_state.mainLLMInput += f"user:\n{userInput.text}\n\n"
            finalOutput = LLMResponse(st.session_state.mainLLMInput, LLM)

        else:
            while not (routerResponse.startswith('{') and routerResponse.endswith('}')):
                routerResponse = LLMResponse(st.session_state.routerLLMInput, LLM)

            articles, URLs = WebSearchToolCallResults(routerResponse)
            summariseArticlesPrompt = summarise_message.format(news=articles)
            articlesSummary = LLMResponse(summariseArticlesPrompt, LLM)

            st.session_state.mainLLMInput += f"user:\n{userInput.text}\n\ncontext:\n{articlesSummary}\n\n"
            finalOutput = LLMResponse(st.session_state.mainLLMInput, LLM)

            finalOutput += "\n\nReferences:\n"
            for i in range(len(URLs)):
                finalOutput += f"   {1+i}. {URLs[i]}\n"

        with output_container:
            for message in st.session_state.messages:
                if message['role'] in ('user', 'human'):
                    with st.chat_message(message["role"], avatar='🧑‍💻'):
                        st.write(message["content"])
                else:
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.write(message["content"])

            st.session_state.messages.append({"role": "user", "content": userInput.text})

            with st.chat_message('user', avatar='🧑‍💻'):
                st.markdown(userInput.text)

            st.session_state.mainLLMInput += f"assistant:\n{finalOutput}\n\n"
            st.session_state.routerLLMInput += f"assistant:\n{finalOutput}\n\n"

            with st.chat_message('ai', avatar="🤖"):
                st.write_stream(stream_response)

            st.session_state.messages.append({"role": "assistant", "content": finalOutput})