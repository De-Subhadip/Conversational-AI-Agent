import os
import time
import json
import warnings
import streamlit as st
from additionals import prompts
from duckduckgo_search import DDGS
from langchain_together import ChatTogether

warnings.filterwarnings("ignore")

system_message, router_message, summarise_message, context_message = prompts() 

### Setting up the LLM: ###
if not os.environ.get("TOGETHER_API_KEY"):
  os.environ["TOGETHER_API_KEY"] = st.secrets['TOGETHER_API_KEY']

LLM = ChatTogether(model="meta-llama/Llama-4-Scout-17B-16E-Instruct")


### Setting up the sesssion state, app title and layout: ###
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.LLM_input = system_message
    st.session_state.routerLLMInput = router_message

st.set_page_config(layout='wide')

_, title_col, _ = st.columns([1.75, 3, 1.25])

with title_col:
    st.title('What\'s on your mind?')

with st.sidebar:
    st.title(':red[Llama 4 Scout ]:robot_face:')
    st.divider()
    clear_chat_history = st.button('**Erase Chatbot Memory**', type='primary')
    if clear_chat_history:
        st.session_state.messages = []
        st.session_state.LLM_input = system_message


### Setting up the Chatbot UI: ###
_, chatbot_UI, _ = st.columns([1, 3, 1], border=False)

def stream_response():
    for token in finalOutput.split(" "):
        yield token + " "
        time.sleep(0.1)

with chatbot_UI:
    output_container = st.container(height=400, border=False)

    if userInput := st.chat_input("Chat with Llama 4 Scout"):
        st.session_state.routerLLMInput += f"User:\n{userInput}\n\n"
        response1 = LLM.invoke(st.session_state.routerLLMInput)
        response1 = str(dict(response1)['content'])

        if not (response1.startswith('{') and response1.endswith('}')):
            st.session_state.LLM_input += f"user:\n{userInput}\n\n"
            finalOutput = LLM.invoke(st.session_state.LLM_input)
            finalOutput = dict(finalOutput)['content']

        else:
            while not (response1.startswith('{') and response1.endswith('}')):
                response1 = LLM.invoke(st.session_state.routerLLMInput)
                response1 = str(dict(response1)['content'])

            response1 = json.loads(response1)
            response1 = response1['params']['query']

            results = DDGS().text(response1, max_results=5)
            articles = ''
            for result in results:
                articles += f"{result['body']}\n\n"

            prompt2 = summarise_message.format(news=articles)
            response2 = LLM.invoke(prompt2)
            response2 = dict(response2)['content']

            st.session_state.LLM_input += f"user:\n{userInput}\n\ncontext:\n{response2}\n\n"
            finalOutput = LLM.invoke(st.session_state.LLM_input)
            finalOutput = dict(finalOutput)['content']

            finalOutput += "\n\nReferences:\n"
            for i in range(len(results)):
                finalOutput += f"   {1+i}. {results[i]['href']}\n"

        with output_container:
            for message in st.session_state.messages:
                if message['role'] in ('user', 'human'):
                    with st.chat_message(message["role"], avatar='🧑‍💻'):
                        st.write(message["content"])
                else:
                    with st.chat_message(message["role"], avatar='🤖'):
                        st.write(message["content"])

            st.session_state.messages.append({"role": "user", "content": userInput})

            with st.chat_message('user', avatar='🧑‍💻'):
                st.markdown(userInput)

            st.session_state.LLM_input += f"assistant:\n{finalOutput}\n\n"
            st.session_state.routerLLMInput += f"assistant:\n{finalOutput}\n\n"

            with st.chat_message('ai', avatar="🤖"):
                st.write_stream(stream_response)

            st.session_state.messages.append({"role": "assistant", "content": finalOutput})