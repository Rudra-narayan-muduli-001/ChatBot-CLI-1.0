![ChatBot CLI Banner](https://capsule-render.vercel.app/api?type=waving&height=200&color=gradient&text=ChatBot%20CLI&fontAlignY=40&desc=Your%20Personal%20AI%20Assistant%20—%20General%20%26%20Realtime%20Chat)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.11+-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20(LLaMA%203.3)-f55036.svg?logo=groq&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/Search-Tavily-1b3d63.svg?logo=tavily&logoColor=white" alt="Tavily"/>
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-8A2BE2.svg" alt="FAISS"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

##  About

A personal **AI assistant backend** with two chat modes, built on FastAPI and powered by **Groq (LLaMA 3.3)** with optional **Tavily** live web search. Conversations are saved to disk, the assistant learns from your personal data files, and it even speaks back with **free Edge TTS** — no extra API key needed.

##  Features

| | Feature | Description |
|---|---------|-------------|
| ⚡ | **General Chat** | Pure LLM responses via Groq — fast, no web search |
| 🌐 | **Realtime Chat** | Groq + live web search (Tavily) for fresh, up-to-date answers |
| 🧠 | **Learning Memory** | Reads your `learning_data/*.txt` + past chats and retrieves relevant context via a FAISS vector store |
| 💬 | **Session Management** | Sessions persist across server restarts; switch between modes mid-conversation |
| 🔑 | **Multi-Key Fallback** | Add several Groq API keys — auto round-robin + fallback on rate limits |
| 🗣️ | **Text-to-Speech** | Server-side TTS via Microsoft Edge (free, no API key) |
| 🛡️ | **Security** | API keys stay in `.env`, never in code; session IDs sanitized against path traversal |

##  Quick Start

> **Prerequisites:** Python 3.9+ and a free [Groq API key](https://console.groq.com).

### 1. Clone the repository

```bash
git clone https://github.com/Rudra-narayan-muduli-001/ChatBot-CLI-1.0.git
cd ChatBot-CLI
```

### 2. Create the environment file

```bash
cp .env.example .env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the server

```bash
python run.py
```

Wait until you see `ChatBot CLI is online and ready!` — the API is live at **http://localhost:8000** (interactive docs at `/docs`).

### 5. Run the CLI client (separate terminal)

Open a **new terminal** — keep the server running — and start chatting:

```bash
python test.py
```

---

##  CLI Usage

Press a key to pick a chat mode, then just type your messages:

| Key | Mode | Description |
|:---:|------|-------------|
| `1` | **General Chat** | Pure LLM, no web search (faster) |
| `2` | **Realtime Chat** | With Tavily live web search |

Both modes share the **same session**, so you can switch anytime without losing context.

### Commands

| Command | Description |
|---------|-------------|
| `1` / `2` | Switch between General and Realtime chat |
| `/history` | View chat history for the current session |
| `/clear` | Start a brand-new session |
| `/quit` / `/exit` | Exit the CLI |

---

##  API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API name + list of all endpoints |
| `GET` | `/health` | Health status of every service |
| `POST` | `/chat` | General chat (pure LLM, no web search) |
| `POST` | `/chat/realtime` | Realtime chat (Tavily web search + Groq) |
| `GET` | `/chat/history/{session_id}` | Full message history for a session |

**Example request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who are you?", "session_id": null}'
```

**Response:**

```json
{
  "response": "I am your personal AI assistant, built to help with anything you need.",
  "session_id": "3f2a8b1c-..."
}
```

> Send the returned `session_id` back on your next request to continue the conversation.

---

##  Configuration (`.env`)

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Primary Groq key ([get one here](https://console.groq.com)) |
| `GROQ_API_KEY_2, _3, ...` | ❌ | — | Extra keys for fallback / round-robin |
| `GROQ_MODEL` | ❌ | `llama-3.3-70b-versatile` | LLM model to use |
| `TAVILY_API_KEY` | ❌* | — | Required only for Realtime web search ([tavily.com](https://tavily.com)) |
| `TTS_VOICE` | ❌ | `en-GB-RyanNeural` | Edge TTS voice (see `edge-tts --list-voices`) |
| `TTS_RATE` | ❌ | `+22%` | TTS speech speed |
| `ASSISTANT_NAME` | ❌ | `Assistant` | Name used in replies |
| `USER_TITLE` | ❌ | — | Optional title to address you by |

\* Realtime mode still works without `TAVILY_API_KEY`, but web search is disabled.

---

##  Project Structure

```
ChatBot CLI/
├── app/                    # FastAPI application
│   ├── main.py             # App entry point + all HTTP endpoints
│   ├── models.py           # Pydantic request/response models
│   ├── services/           # Business logic
│   │   ├── chat_service.py     # Sessions, history, persistence
│   │   ├── groq_service.py     # General chat: Groq LLM + multi-key fallback
│   │   ├── realtime_service.py # Realtime chat: Tavily + Groq
│   │   └── vector_store.py     # FAISS index + retriever
│   └── utils/              # Helpers
│       ├── retry.py            # Exponential-backoff retry
│       └── time_info.py        # Current date/time for the LLM
├── database/               # Created automatically
│   ├── learning_data/      # Personal info (.txt) the assistant learns from
│   ├── chats_data/         # Conversation history (JSON)
│   └── vector_store/       # FAISS index files
├── config.py               # Central configuration (reads .env)
├── run.py                  # Start the server
├── test.py                 # CLI chat client
├── requirements.txt
├── .env.example            # Environment template
└── AGENTS.md
```

##  Tech Stack

- **Backend:** FastAPI + Uvicorn
- **LLM:** Groq — LLaMA 3.3 70B
- **Search:** Tavily API
- **Vector Store:** FAISS + sentence-transformers (`all-MiniLM-L6-v2`)
- **TTS:** edge-tts (Microsoft Edge)
- **CLI:** requests + rich

##  License

Distributed under the **MIT License**.

---

<p align="center">
  <sub>Made with Python, FastAPI and a little bit of magic.</sub>
</p>
