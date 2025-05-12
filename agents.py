from tools import *

def agentRAG(conversation_history, RAG_Query, model, retriever):
    """
    Given a user input, this agent does the following things:
    - performs query transformation based on past conversations
    - retrieves relevant documents from the vector store based on the transformed query
    - checks if the retrieved documents have enough info to answer the transformed query
        - if it has enough info, the agent outputs a "Yes"
        - if it does not have enough info, the agent outputs a "No"
    """


    from prompts import RAGqueryTransformationPrompt, RAGResponsePrompt

    if conversation_history != "":
        RAGqueryTransformationPrompt = RAGqueryTransformationPrompt.format(
            past_conversations = conversation_history,
            query = RAG_Query
        )

        RAG_Query = LLMResponse(RAGqueryTransformationPrompt, LLM=model)

    relevantDocs = retriever.invoke(RAG_Query)

    documents = ""
    for i in range(len(relevantDocs)):
        documentContent = dict(relevantDocs[i])['page_content']
        if documentContent not in documents:
            documents += f"{documentContent}\n\n"

    RAGResponsePrompt = RAGResponsePrompt.format(
        documents = documents,
        question = RAG_Query
    )

    answer = LLMResponse(RAGResponsePrompt, LLM=model).lower()

    return {
        "response" : answer,
        "RAG_Query" : RAG_Query
        }



def agentUseChat(conversation_history, current_input, model):
    """
    This agent checks if the user's current query can be answered by itself or by using the provided past conversation history.
    - if it can answer, output a "yes"
    - else, output a "no"
    """


    from prompts import queryRouterPrompt

    queryRouterPrompt = queryRouterPrompt.format(
        past_conversations = conversation_history,
        question = current_input
    )

    response = LLMResponse(queryRouterPrompt, LLM=model).lower()

    while response not in ("yes", "no"):
        response = LLMResponse(queryRouterPrompt, LLM=model).lower()
    
    return {"response" : response}


def agentChat(conversation_history, model):
    """
    This agent simply answers responds to user inputs based on previous conversations and its own knowledge.
    """


    response = LLMResponse(conversation_history, LLM=model)
    return {"response" : response}


def agentWebSearch(conversation_history, current_input, model):
    """
    This agent does the following:
    - optimizes the user's query for yielding better search results
    - classifies the optimized query for news/text search
    - calls the appropriate search fucntion (news/text) to get the articles and URLs
    - summarizes the articles 
    - checks if images or videos or both would aid in better understanding of the user apart from the textual answer
        - if images are necessary, it returns the article summary, article URLs, and image URLs
        - if videos are necessary, it returns the article summary, article URLs, and video URLs
        - if both are necessary, it returns the article summary, article URLs, image URLs, and video URLs
        - if nothing is necessary, it returns just the article summary and URLs 
    """


    import time
    from prompts import searchQueryOptimizationPrompt, searchQueryClassificationPrompt, articleSummarizationPrompt, searchResponsePrompt, mediaOutputPrompt

    searchQueryOptimizationPrompt = searchQueryOptimizationPrompt.format(
        past_conversations = conversation_history,
        query = current_input
    )

    optimized_Search_Query = LLMResponse(searchQueryOptimizationPrompt, LLM=model)

    searchQueryClassificationPrompt = searchQueryClassificationPrompt.format(query = optimized_Search_Query)
    search_Query_Type = LLMResponse(searchQueryClassificationPrompt, LLM=model).lower()

    while search_Query_Type not in ("news search", "text search"):
        search_Query_Type = LLMResponse(searchQueryClassificationPrompt, LLM=model).lower()

    if search_Query_Type == "news search":
        articles, URLs = news_search(optimized_Search_Query)
    elif search_Query_Type == "text search":
        articles, URLs = text_search(optimized_Search_Query)

    time.sleep(5)

    imgURLS = image_search(optimized_Search_Query)

    time.sleep(5)

    vidURLs = video_search(optimized_Search_Query)

    articleSummarizationPrompt = articleSummarizationPrompt.format(articles = articles)
    articles_summary = LLMResponse(articleSummarizationPrompt, LLM=model)

    searchResponsePrompt = searchResponsePrompt.format(
        article = articles_summary,
        past_conversations = conversation_history,
        query = current_input
    )

    search_Response = LLMResponse(searchResponsePrompt, LLM=model)


    return {
        "response" : search_Response,
        "sources" : URLs,
        "images" : imgURLS,
        "videos" : vidURLs
    }

"""
    mediaOutputPrompt = mediaOutputPrompt.format(
        query = optimized_Search_Query
    )

    media = LLMResponse(mediaOutputPrompt, LLM=model)

    while media not in ("images", "videos", "both", "none"):
        media = LLMResponse(mediaOutputPrompt, LLM=model)

    if media == "images":
        imgURLS = image_search(optimized_Search_Query)
        return {
            "response" : search_Response,
            "URLs" : URLs,
            "images" : imgURLS
        }
    elif media == "videos":
        vidURLs = video_search(optimized_Search_Query)
        return {
            "response" : search_Response,
            "URLs" : URLs,
            "videos" : vidURLs
        }
    elif media == "both":
        imgURLS = image_search(optimized_Search_Query)
        vidURLs = video_search(optimized_Search_Query)
        return {
            "response" : search_Response,
            "URLs" : URLs,
            "images" : imgURLS,
            "videos" : vidURLs
        }
    else:
        return {
            "response" : search_Response,
            "URLs" : URLs
        }
"""