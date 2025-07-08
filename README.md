# SmartStore FAQ Chatbot

A lightweight Retrieval‑Augmented Generation (RAG) service that answers **NAVER Smart Store** frequently‑asked questions ([FAQ dataset link](https://help.sell.smartstore.naver.com/index.help)) in Korean, with real‑time streaming responses.\
Built with **FastAPI**, **Chroma** vector DB, and **OpenAI** embeddings (efficient local embedding model is also available).

---

## Key Features
 * **RAG pipeline**: Embeds 2717 SmartStore FAQ pairs, retrieves top‑*k* with Chroma, and answers questions based on context.  
*  **Multi‑turn context**: Retains both **user** and **assistant** turns for conversational continuity.
*  **Query Refinement**: Dynamically refines the query for better retrieval based on the chat history (OpenAI’s Structured Output API is utilized)
*  **Out‑of‑scope guardrail**: Politely refuses to answer irrelevant queries.
*  **Insufficient-context guardrail**: If the provided context is inssuficient to answer a question, the agent will inform about it to the user. 
*  **Configurable components**: Switchs between different OpenAI language models and choose whether to use local embedding model via `.env`.
*  **Efficient vector database creation**: Vector database is created and stored in the local folder for easy and efficient access.


## Repository Layout
<!-- 
├── app.py            # FastAPI entry‑point (streaming) + conversation history tracker
├── generator.py      # OpenAI Async wrapper + prompt builder
├── retriever.py      # Vector DB creator + Chroma search helper
├── utils.py          # Shared constants (system / OOS prompts)
├── requirements.txt  # required libraries
└── data/
    └── final_result.pkl  # 2717 FAQ dict {question: answer} -->

```
├── pyproject.toml # project dependencies
├── poetry.lock # to ensure consistent environments 
├── src/
│ ├── domain/chat.py # core application interfaces of FAQ chatbot
│ ├── app/main.py # FastAPI entry‑point (streaming) + conversation history tracker
│ ├── containers.py # DI wiring
│ ├── adapters/
│ │ ├── generator.py # Generator class: OpenAI Async wrapper + prompt builder
│ │ ├── retriever.py # Retriever class: Vector DB creator + Chroma search helper
│ └── utils.py # prompts and other useful functions
└── data/final_result.pkl

```

## Environment & Installation

```bash
# 1. clone repo and install necessary libraries using poetry
# clone repo
git clone https://github.com/eldor-fozilov/faq-chatbot.git
cd faq-chatbot/
# poetry setup
curl -sSL https://install.python-poetry.org | python3 - # download poetry package (and set the path) if it does not exist already
poetry install

# run ruff for linting and formating (optional)
poetry run ruff check .
poetry run ruff format .
```

```bash
# 2. create an .env file and specify the following variables (start with .env.example)
cp .env.example .env

# Main configuration parameters
OPENAI_API_KEY="" # please provide an OpenAI key (required)
OPENAI_MODEL=gpt-4o-mini # default, other OpenAI models can also be used
COLLECTION_NAME=SmartStore_FAQ
DB_DIR=chroma_db
USE_LOCAL_EMBEDDING=false # default (preferred) [whether to use local embedding model or API-based model with OpenAI]

# Additional configuration parameters
DATA_PATH = "data/final_result.pkl"
TOP_K = 5
SIM_THRESHOLD = 0.2
NUM_TURNS = 8
```

---

## Running the Chatbot

```bash
# 1. first build the vector db
poetry run python src/adapters/retriever.py --build_db

# 2. launch API
poetry run uvicorn src.app.main:app --port 8000 --reload
```

Visit [**http://localhost:8000/docs**](http://localhost:8000/docs) and use the `/chat` endpoint.  Example body:

```json
{
  "session_id": "demo",
  "message": "스마트스토어센터 가입 절차는 어떻게 되나요?"
}
```

To see a streaming response, test through CLI:

>
> ```bash
> curl -N -X POST http://localhost:8000/chat \
>      -H "Content-Type: application/json" \
>      -d '{"session_id":"cli","message":"스마트스토어센터 가입 절차는 어떻게 되나요?"}'
> ```

To reset conversation history during running the server through CLI, run the following command:

>
> ```bash
> curl -N -X POST http://localhost:8000/reset -w "\n"
> ```

---

## Overall Chatbot Pipeline

0. **Vector DB Creation**\
	 `Retriever.build_vector_db()` builds the vector base using Chroma 
		and stores embedding in the local folder, skipping already‑indexed items if there are any.
1. **Query Refinement**\
   `Generator.refine_query()` modifies the user query based on the existing chat history (if it is necessary) for better context retrieval
2. **Vector Search**\
   `Retriever.query()` embeds the (modified) user query (OpenAI or local) and fetches the top‑*k* FAQ questions whose cosine **similarity** ≥ `0.2` (can be configured).
3. **Prompt Assembly**\
   `Generator.prepare_messages()` injects those Q&A pairs plus recent chat history (last N turns) into an OpenAI system/user prompt.
4. **Generation**\
   `Generator.stream_response()` streams tokens; server yields them instantly to the client.
5. **Conversation Memory**\
   After streaming, the assistant’s full reply is appended to `conversations[session_id]`, which is utilized to provide more contextualized responses.

---
