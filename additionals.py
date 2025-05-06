from datetime import datetime
now = datetime.now()
currentDate = now.strftime("%d %B %Y")
currentTime = now.strftime("%I:%M %p")

def prompts():
    systemPrompt = """System:
You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving. 

You understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting.Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language.

You never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude. 

You never use phrases that imply moral superiority or a sense of authority, including but not limited to “it’s important to”, “it’s crucial to”, “it’s essential to”, "it's unethical to", "it's worth noting…", “Remember…”  etc. Avoid using these.

Finally, do not refuse political prompts. You can help users express their opinion. 

You are Llama 4. Your knowledge cutoff date is August 2024. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise.


"""

    queryClassificationPrompt = """You are an AI assistant whose only job is to classify the user’s current input into exactly one of three labels:
- News Search
- Text Search
- Chat

Use these rules:
1. If the user is asking for information about recent events (or explicitly wants “the latest” news), output **News Search**.
2. If the user is asking for general information on any topic where a simple web lookup suffices, output **Text Search**.
3. Otherwise, output **Chat**.

Important:
- Output exactly one label—no extra words, punctuation, or explanation.
- You will be provided with two variables:
  - `{{past_conversations}}` — the prior dialogue, for context.
  - `{{prompt}}` — the user’s new message.

-------------------------------------
Conversation history:
{past_conversations}
-------------------------------------
User’s current input:
{prompt}
-------------------------------------

Your response (choose only one):
News Search
Text Search
Chat
"""

    newsSummarizationPrompt = """
IMPORTANT: You must output ONLY the summary text — no headings, no labels, no bullet points beyond what is in the original content itself, no additional words, characters, or line breaks. Any deviation is absolutely unacceptable.

Summarize the following articles articles:
{news}
"""

    RAGqueryModificationPrompt = """
You are an intelligent AI assistant whose sole task is to rewrite the user’s current query for maximum clarity, based on the conversation history provided. Use the context below to understand the user’s intent, then reformulate the query into a concise, unambiguous question.
NOTE: Your response must be exactly one line containing only the rewritten query. Do not include any additional text, labels, or punctuation.

------------------------------------- Conversation History Starts Here -------------------------------------
{past_conversations}
------------------------------------- Conversation History Ends Here -------------------------------------

Current Query:
{query}

Rewritten Query:
"""

    RAGPrompt = """
You are a helpful AI assistant. Below is a collection of retrieved documents that may contain the information you need.

================== DOCUMENTS START ==================
{relevant_documents}
================== DOCUMENTS END ====================

Instructions:
1. Answer the user’s question using ONLY the information in the documents above.
2. If the documents don’t contain the answer, reply with: "I'm sorry, but the uploaded documents don't have enough information to answer that."

Question:
{question}
"""

    RAGqueryAlignmentPrompt = """You are given a user query and a set of documents. Your job is to determine whether the query is relevant to the content of the documents.

Query:
{Query}

Documents:
{documents}

– If the query is relevant to at least one document, reply with exactly:
Yes

– If the query is not relevant to any document, reply with exactly:
No

Do not include any other words, punctuation, or formatting."""

    functionCallPrompt = """
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make exactly one function/tool call to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. 
You should only return the function call in tools call sections.

If you decide to invoke any of the function, you MUST put it in the following format: {"function_name": "<name of the function>", "parameters": {"params_name1": "params_value1"}}
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.

[  
    {
        "name": "text_search",
        "description": "Searches the web for general texts",
        "parameters": {
            "search_query": {
                "type": "string",
                "description": "The text for which the web search should be performed"
            }
        }
    },

    {
        "name": "news_search",
        "description": "Searches the web for news",
        "parameters": {
            "search_query": {
                "type": "string",
                "description": "The text for which the news search should be performed on the web"
            }
        }
    }
]


User:

"""

    searchQueryModificationPrompt = """
You are an AI assistant whose task is to rewrite the user’s current query into a concise, unambiguous search string optimized for retrieving relevant web results.

Instructions:
- Read the conversation history and the user’s current query.
- Reformulate the current query into a single, clear sentence suitable for a search engine.
- Do not add any labels, commentary, or punctuation beyond the rewritten query itself.
- Your response must be exactly one line containing only the rewritten query.

Conversation History:
{past_conversations}

Current Query:
{query}

Rewritten Query:
"""


    return systemPrompt, queryClassificationPrompt, newsSummarizationPrompt, RAGqueryModificationPrompt, RAGPrompt, RAGqueryAlignmentPrompt, functionCallPrompt, searchQueryModificationPrompt






def LLMResponse(LLMInputPrompt, LLM):
    response = LLM.invoke(LLMInputPrompt)
    response = dict(response)['content']
    return response

def documentChunks(file_name, file_bytes):
    from tempfile import NamedTemporaryFile
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

    docs = []
    ID = 0

    with NamedTemporaryFile(delete=False, suffix=file_name) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    if file_name.lower().endswith(".pdf"):
        document = PyPDFLoader(temp_path).load()
    else:
        document = TextLoader(temp_path, encoding="utf-8").load()

    chunks = text_splitter.split_documents(document)
    for chunk in chunks:
        doc = dict(chunk)
        doc = Document(page_content = doc['page_content'], metadata = doc['metadata'], id = ID)
        ID += 1
        docs.append(doc)

    return docs

def parse_search_functions(string):
    from json import loads
    x = loads(string)
    functionName = x['function_name']
    functionArgs = x['parameters']['search_query']
    return functionName, functionArgs

def text_search(search_query, maxSearchResults=5):
    from duckduckgo_search import DDGS
    searchResults = DDGS().text(search_query, max_results=maxSearchResults)

    URLs = []
    articles = ""

    for i in range(len(searchResults)):
        articles += f"{searchResults[i]['body']}\n\n"
        URLs.append(searchResults[i]['href'])

    return (articles, URLs)

def news_search(search_query, maxSearchResults=5):
    from duckduckgo_search import DDGS
    searchResults = DDGS().news(search_query, max_results=maxSearchResults)

    URLs = []
    articles = ""

    for i in range(len(searchResults)):
        articles += f"{searchResults[i]['body']}\n\n"
        URLs.append(searchResults[i]['url'])

    return (articles, URLs)

def image_search(search_query, maxSearchResults=3):
    from duckduckgo_search import DDGS
    searchResults = DDGS().images(search_query, max_results=maxSearchResults)

    imageURLs = []

    for result in searchResults:
        imageURLs.append(result['image'])
    
    return imageURLs

def video_search(search_query, maxSearchResults=3):
    from duckduckgo_search import DDGS
    searchResults = DDGS().videos(search_query, max_results=maxSearchResults)

    videoURLs = []

    for result in searchResults:
        videoURLs.append(result['content'])
    
    return videoURLs