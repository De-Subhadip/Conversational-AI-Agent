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