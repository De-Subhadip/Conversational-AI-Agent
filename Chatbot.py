import os
import warnings
import langsmith
from tools import *
from agents import *
from prompts import *
import streamlit as st
from uuid import uuid4
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_together import ChatTogether, TogetherEmbeddings

warnings.filterwarnings("ignore")



### ----------- Setting up the LLM and Tracing: ----------- ###
if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Chat Bot"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

LLM = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")



### ----------- Session states: ----------- ###
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.uploadedFileNames = []
    st.session_state.Conversations = systemPrompt
    st.session_state.RAGConversations = ""



### ----------- App layout: ----------- ###
st.set_page_config(layout='wide')

st.markdown("<h2 style='text-align: center;'>What\'s on your mind?</h2>", unsafe_allow_html=True)

st.markdown("""<style> [data-testid="stVerticalBlock"] h2 { margin-top: -60px; } </style>""", unsafe_allow_html=True)

# st.markdown("""<style> .stChatInput div { min-height: 50px; } </style>""", unsafe_allow_html=True)



### ----------- Vector Store and Retriever: ----------- ###
@st.cache_resource
def get_vector_store():
    embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    vectorStore = InMemoryVectorStore(embedding_model)
    retriever = vectorStore.as_retriever(search_kwargs = {'k':5})
    return vectorStore, retriever

vectorStore, retriever = get_vector_store()



### ----------- Sidebar: ----------- ###
with st.sidebar:
    @st.fragment(run_every="1s")
    def DateTimeFragment():
        import pytz
        from datetime import datetime, timezone
        tz = pytz.timezone(st.context.timezone)
        now = datetime.now(timezone.utc)
        currentDate = now.astimezone(tz).strftime("%d %B %Y")
        currentTime = now.astimezone(tz).strftime("%I:%M:%S %p")
        st.metric(label=":red[**Current Date:**]", value=currentDate)
        st.metric(label=":red[**Current Time:**]", value=currentTime)

    DateTimeFragment()

    st.divider()

    if st.button("**Clear Agent's Memory**", type='primary'):
        st.session_state.messages = []
        st.session_state.uploadedFileNames = []
        st.session_state.Conversations = systemPrompt
        st.session_state.RAGConversations = ""
        vectorStore.delete()

    st.divider()

    RAG = st.toggle(":red-background[**Use RAG**]", key="RAG")
    if st.session_state.RAG:
        uploaded_files = st.file_uploader(label=" ", accept_multiple_files=True, type=["pdf", "txt"])

        ### ----------- Ingesting documents, if any: ----------- ###
        if uploaded_files is not None:
            for file in uploaded_files:
                fileBytes = file.read()
                fileName = file.name
                if fileName not in st.session_state.uploadedFileNames:
                    st.session_state.uploadedFileNames.append(fileName)
                    with st.spinner(f"Ingesting {fileName} ...", show_time=True):
                        docs = documentChunks(file_name=fileName, file_bytes=fileBytes)
                        uuIDs = [str(uuid4()) for _ in range(len(docs))]
                        vectorStore.add_documents(documents=docs, ids=uuIDs)
            
            st.subheader("**Uploaded Documents:**")
            for i, filename in enumerate(st.session_state.uploadedFileNames):
                st.write(f"{i+1}. {filename}")



### ----------- Streaming Response: ----------- ###
def stream_response():
    from time import sleep
    for token in finalOutput.split(" "):
        yield token + " "
        sleep(0.1)



### ----------- Chatbot UI: ----------- ###
_, chatUI, _ = st.columns([1, 5, 1], border=False)

with chatUI:
    output_container = st.container(height=480, border=False)

    if userInput := st.chat_input("What do you want to ask?"):


        ### ----------- If RAG is enabled: ----------- ###
        if st.session_state.RAG:
            if len(vectorStore.store) != 0: # If vector store is not empty
                result_dict = agentRAG(
                    conversation_history = st.session_state.RAGConversations,
                    RAG_Query = userInput,
                    model = LLM,
                    retriever = retriever
                )
                st.session_state.RAGConversations += f"user: \n{result_dict["RAG_Query"]} \n\n"
                finalOutput = result_dict["response"]

            else: # If vector store is empty
                st.session_state.RAGConversations += f"user: \n{userInput} \n\n"
                finalOutput = "No uploaded documents found."


        ### ----------- If RAG is disabled: ----------- ###
        else:
            st.session_state.Conversations += f"user: \n{userInput} \n\n"
            use_Chat = agentUseChat(
                conversation_history = st.session_state.Conversations,
                current_input = userInput,
                model = LLM
            )

            if use_Chat["response"] == "yes":
                finalOutput = agentChat(conversation_history = st.session_state.Conversations, model=LLM)
                finalOutput = finalOutput["response"]
            
            else: # use_Chat["response"] == "no":
                result_dict = agentWebSearch(
                    conversation_history = st.session_state.Conversations,
                    current_input = userInput,
                    model = LLM
                )
                finalOutput = f"{result_dict["response"]} \n\nSources: \n"
                for i in range(len(result_dict["sources"])):
                    finalOutput += f"   {1+i}. {result_dict["sources"][i]}\n"



        ### ----------- The Output: ----------- ###
        with output_container:
            for message in st.session_state.messages:
                if message['role'] in ('user', 'human'):
                    with st.chat_message(message["role"], avatar='🧑‍💻'):
                        st.markdown(message["content"])

                elif message['role'] in ('ai', 'assistant'):
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.markdown(message["content"])

            st.session_state.messages.append({"role": "user", "content": userInput})

            with st.chat_message('user', avatar='🧑‍💻'):
                st.markdown(userInput)

            if st.session_state.RAG:
                st.session_state.RAGConversations += f"assistant:\n{finalOutput}\n\n"
            else:
                st.session_state.Conversations += f"assistant:\n{finalOutput}\n\n"

            with st.chat_message('ai', avatar="🤖"):
                st.write_stream(stream_response)

            st.session_state.messages.append({"role": "assistant", "content": finalOutput})