"""
FASTAPI APPLICATION PACKAGE

This directory is the main Python package for the ChatBot CLI backend.
The presence of __init__.py makes Python treat 'app' as a package, so you can:

    from app.main import app
    from app.models import ChatRequest
    from app.services.chat_service import ChatService

FILE STRUCTURE:
  app/
    __init__.py     - This file marks app as a package.
    main.py         - FastAPI app and all HTTP endpoints.
    models.py       - Pydantic models for API requests and chat storage.
    services/       - Business logic: chat sessions, Groq LLM, realtime, vector store.
    utils/          - Helpers: retry with backoff, current date/time.
"""
