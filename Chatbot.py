import os
import warnings
import additionals
import streamlit as st
from uuid import uuid4
from additionals import *
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_together import ChatTogether, TogetherEmbeddings

warnings.filterwarnings("ignore")

system_message, queryClassificationPrompt, summarise_message, RAGqueryModificationPrompt, RAGPrompt, RAGqueryAlignmentPrompt, functionCallPrompt, searchQueryModificationPrompt = prompts() 

### Setting up the LLM: ###
if not os.environ.get("TOGETHER_API_KEY"):
  os.environ["TOGETHER_API_KEY"] = st.secrets['TOGETHER_API_KEY']

LLM = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")

### Setting up the sesssion state, app title and layout: ###
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.uploadedFileNames = []
    st.session_state.pastConversations = ""
    st.session_state.RAGContext = ""
    st.session_state.RAGConversations = ""

st.set_page_config(layout='wide')

st.markdown("<h2 style='text-align: center;'>What\'s on your mind?</h2>", unsafe_allow_html=True)

st.markdown("""<style> [data-testid="stVerticalBlock"] h2 { margin-top: -60px; } </style>""", unsafe_allow_html=True)

st.markdown("""<style> .stChatInput div { min-height: 50px; } </style>""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    embedding_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    vectorStore = InMemoryVectorStore(embedding_model)
    retriever = vectorStore.as_retriever(search_kwargs = {'k':5})
    return vectorStore, retriever

vectorStore, retriever = get_vector_store()


### ---------------------- Side Bar: ---------------------- ###
with st.sidebar:
    @st.fragment(run_every="1s")
    def dateFragment():
        from datetime import datetime
        currentTime = datetime.now().strftime("%d %B %Y")
        st.metric(label=":red[**Current Date:**]", value=currentTime)
    dateFragment()

    @st.fragment(run_every="1s")
    def timeFragment():
        from datetime import datetime
        currentTime = datetime.now().strftime("%I:%M:%S %p")
        st.metric(label=":red[**Current Time (GMT):**]", value=currentTime)
    timeFragment()

    st.divider()

    if st.button("**Clear Agent's Memory**", type='primary'):
        st.session_state.messages = []
        st.session_state.uploadedFileNames = []
        st.session_state.pastConversations = ""
        st.session_state.RAGContext = ""
        st.session_state.RAGConversations = ""

    st.divider()

    useRAG = st.toggle(":red[**Use RAG**]", key="useRAG")
    if useRAG:
        st.write("**Uploaded File Names:**")
        for i, filename in enumerate(st.session_state.uploadedFileNames):
            st.write(f"{i+1}. {filename}")



### Setting up the Chatbot UI: ###
imgContainer, chatContainer, vidContainer = st.columns([1, 5, 1], border=False)

def stream_response():
    from time import sleep
    for token in finalOutput.split(" "):
        yield token + " "
        sleep(0.1)


with chatContainer:
    output_container = st.container(height=480, border=False)

    if userInput := st.chat_input("Chat with Llama 4", accept_file=True, file_type=['pdf', 'txt']):
        if st.session_state.useRAG:
            RAGqueryModificationPrompt = RAGqueryModificationPrompt.format(past_conversations=st.session_state.RAGConversations, query=userInput.text)
            modifiedQuery = LLMResponse(RAGqueryModificationPrompt, LLM)

            if userInput["files"]:
                with st.spinner("Ingesting Documents...", show_time=True):
                    for file in userInput["files"]:
                        fileBytes = file.read()
                        fileName = file.name
                        st.session_state.uploadedFileNames.append(fileName)
                        docs = documentChunks(file_name=fileName, file_bytes=fileBytes)
                        
                    uuIDs = [str(uuid4()) for _ in range(len(docs))]
                    vectorStore.add_documents(documents=docs, ids=uuIDs)

            if len(vectorStore.store) != 0:
                relevantDocs = retriever.invoke(userInput.text)
                contexts = ""
                for i in range(1, len(relevantDocs)+1):
                    documentContent = dict(relevantDocs[i-1])['page_content']
                    if documentContent not in st.session_state.RAGContext:
                        st.session_state.RAGContext += f"{documentContent}\n\n"

                RAGPrompt = RAGPrompt.format(relevant_documents = st.session_state.RAGContext, question = modifiedQuery)
                RAGResponse = LLMResponse(RAGPrompt, LLM)

                if RAGResponse == "I'm sorry, but the uploaded documents don't have enough information to answer that.":
                    RAGqueryAlignmentPrompt = RAGqueryAlignmentPrompt.format(Query=modifiedQuery, documents=st.session_state.RAGContext)
                    queryAlignment = LLMResponse(RAGqueryAlignmentPrompt, LLM)

                    if queryAlignment == 'No':
                        st.session_state.RAGConversations += f"User:\n{modifiedQuery}\n\n"
                        finalOutput = "Your query does not align with the contents of the uploaded documents."
                    else:
                        #######################################################

                        queryClassificationPrompt = queryClassificationPrompt.format(past_conversations=st.session_state.RAGConversations, prompt=modifiedQuery)
                        classifiedQuery = LLMResponse(queryClassificationPrompt, LLM).lower()
                        
                        if classifiedQuery == "chat":
                            st.session_state.RAGConversations += f"User:\n{modifiedQuery}\n\n"
                            inputPrompt = system_message + st.session_state.RAGConversations
                            finalOutput = LLMResponse(inputPrompt, LLM)
                        else:
                            searchQueryModificationPrompt = searchQueryModificationPrompt.format(past_conversations=st.session_state.RAGConversations, query=modifiedQuery)
                            modifiedQuery = LLMResponse(searchQueryModificationPrompt, LLM)
                            
                            functionCallPrompt += f"{modifiedQuery}\n\nAssistant:"
                            functionToCall = LLMResponse(functionCallPrompt, LLM)
                            functionName, search_query = parse_search_functions(functionToCall)
                            functionName = getattr(additionals, functionName)

                            articles, URLs = functionName(search_query)
                            summariseArticlesPrompt = summarise_message.format(news=articles)
                            articlesSummary = LLMResponse(summariseArticlesPrompt, LLM)

                            st.session_state.RAGConversations += f"User:\n{modifiedQuery}\n\nArticle found on web for answering the above query:\n{articlesSummary}\n\n"

                            finalOutput = LLMResponse(st.session_state.RAGConversations, LLM)
                            finalOutput += "\n\nSources:\n"
                            for i in range(len(URLs)):
                                finalOutput += f"   {1+i}. {URLs[i]}\n"

                        ############################################################
                else:
                    st.session_state.RAGConversations += f"User:\n{modifiedQuery}\n\n"
                    finalOutput = RAGResponse

                st.session_state.RAGConversations += f"Assistant:\n{finalOutput}\n\n"
            else:
                finalOutput = "Did not find any uploaded documents. Please upload some relevant ones."

        else:
            queryClassificationPrompt = queryClassificationPrompt.format(past_conversations=st.session_state.pastConversations, prompt=userInput.text)
            classifiedQuery = LLMResponse(queryClassificationPrompt, LLM).lower()
            
            if classifiedQuery == "chat":
                st.session_state.pastConversations += f"User:\n{userInput.text}\n\n"
                inputPrompt = system_message + st.session_state.pastConversations
                finalOutput = LLMResponse(inputPrompt, LLM)
            else:
                searchQueryModificationPrompt = searchQueryModificationPrompt.format(past_conversations=st.session_state.pastConversations, query=userInput.text)
                modifiedQuery = LLMResponse(searchQueryModificationPrompt, LLM)
                
                functionCallPrompt += f"{modifiedQuery}\n\nAssistant:"
                functionToCall = LLMResponse(functionCallPrompt, LLM)
                functionName, search_query = parse_search_functions(functionToCall)
                functionName = getattr(additionals, functionName)

                articles, URLs = functionName(search_query)
                summariseArticlesPrompt = summarise_message.format(news=articles)
                articlesSummary = LLMResponse(summariseArticlesPrompt, LLM)

                st.session_state.pastConversations += f"User:\n{userInput.text}\n\ncontext:\n{articlesSummary}\n\n"

                finalOutput = LLMResponse(st.session_state.pastConversations, LLM)
                finalOutput += "\n\nSources:\n"
                for i in range(len(URLs)):
                    finalOutput += f"   {1+i}. {URLs[i]}\n"


        with output_container:
            for message in st.session_state.messages:
                if message['role'] in ('user', 'human'):
                    with st.chat_message(message["role"], avatar='🧑‍💻'):
                        st.markdown(message["content"])

                elif message['role'] in ('ai', 'assistant'):
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.markdown(message["content"])

                elif message['role'] == "image":
                    _, imgCol, _ = st.columns([1.25, 3, 1])
                    with imgCol:
                        st.image(message['content'], width=400)

                elif message['role'] == 'video':
                    _, vidCol, _ = st.columns([1.25, 3, 1])
                    with vidCol:
                        st.video(message['content'])

            st.session_state.messages.append({"role": "user", "content": userInput.text})

            with st.chat_message('user', avatar='🧑‍💻'):
                st.markdown(userInput.text)

            st.session_state.pastConversations += f"assistant:\n{finalOutput}\n\n"

            with st.chat_message('ai', avatar="🤖"):
                st.write_stream(stream_response)

            st.session_state.messages.append({"role": "assistant", "content": finalOutput})