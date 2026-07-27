from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEYS, JARVIS_SYSTEM_PROMPT, GROQ_MODEL
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")


def escape_curly_braces(text: str) -> str:
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


def _mask_api_key(key: str) -> str:
    if not key or len(key) <= 12:
        return "***masked***"
    return f"{key[:8]}...{key[-4:]}"


class GroqService:
    _shared_key_index = 0
    _lock = None

    def __init__(self, vector_store_service: VectorStoreService):
        if not GROQ_API_KEYS:
            raise ValueError(
                "No GROQ API keys configured. Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in .env"
            )

        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.8,
            )
            for key in GROQ_API_KEYS
        ]
        self.vector_store_service = vector_store_service
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s)")

    def _invoke_llm(
            self,
            prompt: ChatPromptTemplate,
            messages: list,
            question: str,
    ) -> str:
        n = len(self.llms)
        start_i = GroqService._shared_key_index % n
        current_key_index = GroqService._shared_key_index
        GroqService._shared_key_index += 1

        masked_key = _mask_api_key(GROQ_API_KEYS[start_i])
        logger.info(f"Using API key #{start_i + 1}/{n} (round-robin index: {current_key_index}): {masked_key}")

        last_exc = None
        keys_tried = []
        for j in range(n):
            i = (start_i + j) % n
            keys_tried.append(i)
            try:
                chain = prompt | self.llms[i]
                response = chain.invoke({"history": messages, "question": question})
                if j > 0:
                    masked_success_key = _mask_api_key(GROQ_API_KEYS[i])
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} succeeded: {masked_success_key}")
                return response.content
            except Exception as e:
                last_exc = e
                masked_failed_key = _mask_api_key(GROQ_API_KEYS[i])
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_failed_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_failed_key} - {str(e)[:100]}")
                if n > 1:
                    continue
                raise Exception(f"Error getting response from Groq: {str(e)}") from e

        masked_all_keys = ", ".join([_mask_api_key(GROQ_API_KEYS[i]) for i in keys_tried])
        logger.error(f"All API keys failed. Tried keys: {masked_all_keys}")
        raise Exception(f"Error getting response from Groq: {str(last_exc)}") from last_exc

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        try:
            context = ""
            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            except Exception as retrieval_err:
                logger.warning("Vectore store retrieval failed, using empty context: %s", retrieval_err)

            time_info = get_time_information()
            system_message = JARVIS_SYSTEM_PROMPT + f"\n\nCurrent time and date: {time_info}"
            if context:
                system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escape_curly_braces(context)}"

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

            return self._invoke_llm(prompt, messages, question)
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e
