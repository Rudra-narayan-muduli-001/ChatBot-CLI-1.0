from typing import List, Optional
from tavily import TavilyClient
import logging
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


logger = logging.getLogger(__name__)


class RealtimeGroqService(GroqService):

    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")

    def search_tavily(self, query: str, num_results: int = 5) -> str:
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ""

        try:
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
            logger.info(f"Tavily search completed for query: {query} ({len(results)} results)")
            return ""

    def get_response(self, question: str, chat_history: Optional[List[tuple]] = None) -> str:
        try:
            logger.info(f"Searching Tavily for: {question}")
            search_results = self.search_tavily(question, num_results=5)

            context = ""
            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            except Exception as retrieval_err:
                logger.warning("Vector store retrieval failed, using empty context: %s", retrieval_err)

            time_info = get_time_information()
            system_message = SYSTEM_PROMPT + f"\n\nCurrent time and date: {time_info}"

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

            response_content = self._invoke_llm(prompt, messages, question)
            logger.info(f"Realtime response generated for: {question}")
            return response_content

        except Exception as e:
            logger.error(f"Error in realtime get_response: {e}", exc_info=True)
            raise
