import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Iterator
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService


logger = logging.getLogger("J.A.R.V.I.S")

SAVE_EVERY_IN_CHUNKS = 5

# =========================================================================
# CHAT SERVICE CLASS
# =========================================================================

class ChatService:
    """
    Manages chat sessions: in_memory message lists, load/save to disk, and calling Groq (or Realtime)
    to get replies. All state for active sessions is in self.sessions; saving to disk after each 
    message to conversation survive restarts.

    """

    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """Store references to the Groq and Realtime services ; keep session in memory."""
        self.groq_service = groq_service
        self.realtime_service = realtime_service

        self.sessions: Dict[str, List[ChatMessage]] = {}  # session_id -> ChatHistory

    #-------------------------------------------------------------------------
    # SESSION LOAD / VALIDATE / GET-OR-CREATE
    #-------------------------------------------------------------------------

    def load_session_from_disk(self, session_id: str) -> bool:
        """
        Load a session from database/chat_data/ if a file for this session_id exists.
        
        File name is chat_{safe_session_id}.json, where safe_session_id has dashes/spaces removed.
        On success we put the messages into safe.session[session_id] so later requiests use them.
        Returns True if loaded successfully, False if file missing or unreadable.
        """
        # Sanitize ID for use in filename (no dashes or spaces).
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename

        if not filepath.exists():
            return False
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
                # Convert stored dicts to ChatMessage objects.
            messages = [
                    ChatMessage(role=msg.get("role"), content=msg.get("content"))
                    for msg in chat_dict.get("messages", [])
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning(f"Failed to load chat session {session_id} from disk: {e}")
            return False
        
    def validate_session_id(self, session_id: str) -> bool:
        """
        Return True if session_id is valid and safe to use (non empty, no path traversal, length <= 255).
        Used to reject malicius or invalid IDs before we use them in file paths.
        """
        if not session_id or not session_id.strip():
            return False
        # Reject IDs with path traversal or path separators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Return existing session ID and ensure that session ID exists in memory.

        - If session_id is None, generate a new UUID and return it.
        - If session_id is provided, validate it. If invalid, raise ValueError.
          else try to load from disk. If load fails, create new empty session with this ID.
        Raises ValueError if session_id is invalid (empty, path traversal, too long).
        """
        t0 = time.perf_counter()
        if not session_id:
            new_session_id = str(uuid.uuid4())
            self.sessions[new_session_id] = []
            logger.info("[TIMING] session_get_or_create: %.3fs (new)", time.perf_counter() - t0)
            return new_session_id
        
        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_ID format: {session_id}. Session ID must be non-empty, "
                "not contain path traversal characters, and be <= 255 chars."
                )
        
        if session_id in self.sessions:
            logger.info("[TIMING] session_get_or_create: %.3fs (memory)", time.perf_counter() - t0)
            return session_id
        
        if self.load_session_from_disk(session_id):
            logger.info("[TIMING] session_get_or_create: %.3fs (disk)", time.perf_counter() - t0)
            return session_id
        
        # New session with provided ID (after validation).
        self.sessions[session_id] = []
        logger.info("[TIMING] session_get_or_create: %.3fs (new_id)", time.perf_counter() - t0)
        return session_id
    
    #-------------------------------------------------------------------------
    # MESSAGE HISTORY MANAGEMENT
    #-------------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str):
        """ Append one message (user or assistant) to the session's message list in memory. Create session if it doesn't exist. """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """ Return the list of messages for this session_id, or empty list if session doesn't exist. """
        return self.sessions.get(session_id, [])
    
    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[tuple]:
        """
       Build a list of (user_text, assistant_text) pairs for the LLM prompt.

       We only include complete pairs and cap at MAX_CHAT_HISTORY_TURNS (e.g. 30)
       So the prompt does not grow unbounded. If exclude_last is True, we drop the last user message
       (the current user message that we are about to reply to). 
        """
        messages = self.get_chat_history(session_id)
        history = []
        # If exclude_last, we skip the last message (the current user message we are about to reply to ).
        messages_to_process = messages[:-1] if exclude_last and messages else messages

        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i + 1]
            if user_msg.role == "user" and ai_msg.role == "assistant":
                history.append((user_msg.content, ai_msg.content))
                i += 2
            else:
                i += 1
            # Keep only the most recent turns so the prompt does not exceed token limits.
        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]
        return history
    
    #-------------------------------------------------------------------------
    # PROCESSING MESSAGES (GENERAL AND REALTIME)
    #-------------------------------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one general-chat message : add uses message, call Groq (no web search), add reply, return it.
        """
        logger.info("[GENERAL] Session: %s | User: %.200", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL] History pairs send to llm: %d", len(chat_history))

        response = self.groq_service.get_response(question=user_message, chat_history=chat_history)

        self.add_message(session_id, "assistant", response)
        logger.info("[GENERAL] Response length: %d chars | Preview: %.120s", len(response), response)
        return response
    
    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        """
        Handle one realtime message : add user message , call realtime service (Tavily + Groq), add reply , return it.
        Uses the same session as process_messages so history is shared. Raises valueError if realtime_service is None.
        """
        if not self.realtime_service:
            raise ValueError("Realtime service not initialized. Cannot process realtime queries.")
        logger.info("[REALTIME] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[REALTIME] History pairs send to LLM: %d", len(chat_history))
        response = self.realtime_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        logger.info("[REALTIME] Response length: %d chars | Preview: %.120s", len(response), response)
        return response
    
    def process_message_stream(
            self, session_id: str, user_message: str
    ) -> Iterator[str]:
        
        logger.info("[GENERAL-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)

        self.add_message(session_id, "assistant", "")
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL-STREAM] History pairs sent to LLM: %d", len(chat_history))
        chunk_count = 0
        try:
            for chunk in self.groq_service.stream_response(
                question=user_message, chat_history=chat_history
            ):
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1

                if chunk_count % SAVE_EVERY_IN_CHUNKS == 0:
                    self.save_chat_session(session_id, log_timing=False)
                
                yield chunk
        finally:
            final_response = self.sessions[session_id][-1].content
            logger.info("[GENERAL-STREAM] Completed | Chunks: %d | Response length: %d chars", chunk_count, len(final_response))
            self.save_chat_session(session_id)

    def process_realtime_message_stream(
            self, session_id: str, user_message: str
    ) -> Iterator[str]:
        
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")
        logger.info("[REALTIME-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[REALTIME-STREAM] History pairs sent to LLM: %d", len(chat_history))
        chunk_count = 0

        try:
            for chunk in self.realtime_service.stream_response(
                question=user_message, chat_history=chat_history
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1
                if chunk_count % SAVE_EVERY_IN_CHUNKS == 0:
                    self.save_chat_session(session_id, log_timing=False)
                yield chunk
        finally:
            final_response = self.sessions[session_id][-1].content
            logger.info("[REALTIME-STREAM] Completed | Chunks: %d | Response length: %d chars", chunk_count, len(final_response))
            self.save_chat_session(session_id)

    
    #-------------------------------------------------------------------------
    # PERSIST SESSION TO DISK
    #-------------------------------------------------------------------------

    def save_chat_session(self, session_id: str, log_timing: bool = True):
        """
        Write this session's message so the conversation to database/chat_data/chat_{safe_id}.json.
        
        Called after each message so the conversation is persisted. The vector store
        is rebuilt on startup from these files, so new chats are included after resrart.
        If the session is missing or empty we do nothing. on write error we only log.
        """
        if session_id not in self.sessions or not self.sessions[session_id]:
            return
        
        messages = self.sessions[session_id]
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        chat_dict = {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }

        try:
            t0 = time.perf_counter() if log_timing else 0
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chat_dict, f, ensure_ascii=False, indent=2)
            if log_timing:
                logger.info("[TIMING] save_session_json: %.3fs", time.perf_counter() - t0)
        except Exception as e:
            logger.error(f"Failed to save chat session {session_id} to disk: {e}")
            