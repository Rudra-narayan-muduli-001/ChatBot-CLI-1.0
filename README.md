# J.A.R.V.I.S — AI Chatbot CLI

A personal AI assistant backend with General Chat (pure LLM) and Realtime Chat (with live web search) modes, powered by Groq and Tavily.

## Features

- **General Chat** — Pure LLM responses using Groq (llama-3.3-70b-versatile)
- **Realtime Chat** — LLM responses with live web search via Tavily
- **Session Management** — Conversation history persists across messages
- **Multi-Key Fallback** — Multiple Groq API keys for automatic failover
- **Text-to-Speech** — Uses Microsoft Edge TTS (free, no API key needed)
- **Learning Memory** — Stores personal info in vector store for context-aware replies

## Clone the Repository

```bash
git clone https://github.com/your-username/ChatBot-CLI.git
cd ChatBot-CLI
```

## Setup & Installation

### 1. Create `.env` file

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:
- **`GROQ_API_KEY`** (required) — Get one at [console.groq.com](https://console.groq.com)
- **`TAVILY_API_KEY`** (optional) — Get one at [tavily.com](https://tavily.com) (needed for Realtime Chat)

You can also configure: Groq model, TTS voice/speed, assistant name, and fallback API keys.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
python run.py
```

Wait until you see the server startup message (e.g. `Uvicorn running on http://0.0.0.0:8000`).

### 4. Run the CLI client (in a separate terminal)

Open a **new terminal** without stopping `run.py`:

```bash
python test.py
```

## CLI Usage

When you run `test.py`, select a chat mode first:

| Key | Mode | Description |
|-----|------|-------------|
| `1` | General Chat | Pure LLM, no web search (faster) |
| `2` | Realtime Chat | With Tavily live web search |

Once a mode is selected, type your messages and press Enter to chat.

### Commands

| Command | Description |
|---------|-------------|
| `1` | Switch to General Chat |
| `2` | Switch to Realtime Chat |
| `/history` | View chat history for current session |
| `/clear` | Start a new session |
| `/quit` | Exit the CLI |

You can switch between modes at any time — both modes share the same session.

## Project Structure

```
ChatBot CLI/
├── app/                  # FastAPI application
│   └── main.py           # App entry point
├── database/
│   ├── learning_data/    # Personal info (.txt files)
│   ├── chats_data/       # Conversation history (JSON)
│   └── vector_store/     # FAISS index files
├── config.py             # Central configuration
├── run.py                # Start server
├── test.py               # CLI test client
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── .env                  # Your API keys (not committed)
```
