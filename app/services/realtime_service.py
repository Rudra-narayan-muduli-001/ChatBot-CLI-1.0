"""
REALTIME GROQ SERVICE MODULE
===========================

Extends GroqService to add Tavily web search before calling the LLM. Used by 
ChatService for POST /chat/realtime. Same session and history as general chat;
the only difference is we run a Tavily search for the user's question and add
the results to the system message, then call Groq.

ROUND-ROBIN API KEYS:
  - Shares the same round-robin counter as GroqService (class-level _shared_key_index)
  - This means /chat and /chat/realtime requests use the same rotation sequence.
  - Example: If /chat uses key 1, the next /chat/realtime request will use key 2.
  - All API key usage is logged with masked keys for security and debugging

FLOW:
    1. Search the same Round-Robin counter as GroqService (class-level _shared_key_index)
    2. get_response(question, chat_history): add search results to system message,
        then same as parent: retrieve context from vector store , build prompt, call Groq.

If TAVILY_API_KEY is not set, tavily_client is None and search_tavily returns "";
the user still gets an answer from Groq with no search results.
"""

from typing import List, Optional
from tavily import TavilyClient
import logging 
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import JARVIS_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


logger = logging.getLogger("J.A.R.V.I.S")


# =========================================================================
# REALTIME GROQ SERVICE CLASS  (extends GroqService)
# =========================================================================

class RealtimeGroqService(GroqService):
    """
    Same as GroqService but runs a Tavily web search and adds the results
    to the system message. If Tavily is missing or fails, we still call groq with
    no search results (user gets an answer without realtime data).
    """

    def __init__(self, vector_store_service: VectorStoreService):
        """Call parent init (Groq LLM + vector store); then create Tavily client if TAVILY_API_KEY is set."""
        super().__init__(vector_store_service)
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")

    def search_tavily(self, query: str, num_results: int = 5) -> str:
        """
        Call Tavily API with the given query and return formatted result text for the prompt.
        On any failure (no key, rate limit, network) we return "" so the LLM still gets a reply.
        """
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ""
        
        try:
            # Perform  Tavily search with retries for rate limits and transient errors.
            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=num_results,
                    include_answer=False,
                    include_raw_content=False,
                ),
                max_retries=3,
                initial_delay=1.0,
            )

            results = response.get('results', [])

            if not results:
                logger.warning(f"No Tavily search results found for query: {query}")
                return ""

            # Format search results as text for the system prompt.
            formatted_results = f"Search results for '{query}':\n[start]\n"

            for i, result in enumerate(results[:num_results], 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No description')
                url = result.get('url', '')

                formatted_results += f"Title: {title}\n"
                formatted_results += f"Description: {content}\n"
                if url:
                    formatted_results += f"URL: {url}\n"
                formatted_results += "\n"

            formatted_results += "[end]"
            
            logger.info(f"Tavily search completed for query: {query} ({len(results)}results)")
            return formatted_results
        
        except Exception as e:
            # If search fails (network error, rate limit, etc.), log and return empty.
            # The AI will still respond using its knowledge, just without real-time data.
            logger.info(f"Tavily search completed for query: {query} ({len(results)} results)")
            return ""
        
    def get_response(self, question: str, chat_history: Optional[List[tuple]] = None) -> str:
        """
        Run Tavily search for the questions, add results to the system message, then call Groq
        via the parent's _invoke_llm (same multi-key round-robin and fallback as general chat).
        """
        try:
            logger.info(f"Searching Tavily for: {question}")
            search_results = self.search_tavily(question, num_results=5)

            # Retrieve context from vector store (learning data + past chats).
            # If retrieval fails, use empty context so the LLM still answers (e.g. with Tavily results).
            context = ""
            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            except Exception as retrieval_err:
                logger.warning("Vector store retrieval failed, using empty context: %s", retrieval_err)

            # Build system message: personality + time + Tavily results + retrieved context.
            time_info = get_time_information()
            system_message = JARVIS_SYSTEM_PROMPT + f"\n\nCurrent time and date: {time_info}"

            if search_results:
                escaped_search_results = escape_curly_braces(search_results)
                system_message += f"\n\nRecent search results:\n{escaped_search_results}"

            if context:
                escaped_context = escape_curly_braces(context)
                system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escaped_context}"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            messages = []
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=human_msg))
                    messages.append(AIMessage(content=ai_msg))

            # Uses same round-robin and fallback as general chat: next one-by-one, try next on failure.
            response_content = self._invoke_llm(prompt, messages, question)
            logger.info(f"Realtime response generated for: {question}")
            return response_content
        
        except Exception as e:
            logger.error(f"Error in realtime get_response: {e}", exc_info=True)
            # Re-rise so main.py can return 429 (rate limit) or 500 consistently with general chat
            raise
