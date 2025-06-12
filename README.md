# SmartStore FAQ Chatbot

A lightweight Retrieval‑Augmented Generation (RAG) service that answers **NAVER Smart Store** frequently‑asked questions in Korean, with real‑time streaming responses.\
Built with **FastAPI**, **Chroma** vector DB, and **OpenAI** embeddings (efficient local embedding model is also available).

---

## Key Features
 * **RAG pipeline**: Embeds 2717 SmartStore FAQ pairs, retrieves top‑*k* with Chroma, and answers questions based on context.  
*  **Multi‑turn context**: Retains both **user** and **assistant** turns for conversational continuity.
*  **Out‑of‑scope guardrail**: Politely refuses to answer irrelevant queries.
*  **Insufficient-context guardrail**: If the provided context is inssuficient to answer a question, the agent will inform about it to the user. 
*  **Configurable components**: Switch between different OpenAI language models and choose whether to use local embedding model via `.env`.
*  **Efficient vector database creation**: `retriever.build_vector_db()` creates and stores embedding in the local folder, skipping already‑indexed items if there are any.


## Repository Layout

```
├── app.py            # FastAPI entry‑point (streaming) + conversation history tracker
├── generator.py      # OpenAI Async wrapper + prompt builder
├── retriever.py      # Vector DB creator + Chroma search helper
├── utils.py          # Shared constants (system / OOS prompts)
├── requirements.txt  # required libraries
└── data/
    └── final_result.pkl  # 2717 FAQ dict {question: answer}
```

## Environment & Installation

```bash
# 1. clone repo and install necessary libraries
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. create an .env file and specify the following variables
OPENAI_API_KEY="" # please provide an OpenAI key (required)
OPENAI_MODEL=gpt-4o-mini # default, other OpenAI models can also be used
COLLECTION_NAME=SmartStore_FAQ
DATA_PATH=data/final_result.pkl
DB_DIR=chroma_db
USE_LOCAL_EMBEDDING=false # default (preferred) [whether to use local embedding model or API-based model with OpenAI]
```

---

## Running the Chatbot

```bash
# 1. first build the vector db
python retriever.py --build_db

# 2. launch API
uvicorn app:app --port 8000 --reload
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
> curl -N -X POST http://localhost:8000/reset
> ```

---

## Overall Chatbot Pipeline

0. **Vector DB Creation**\
	 `Retriever.build_vector_db()` builds the vector base using Chroma 
		and stores embedding in the local folder, skipping already‑indexed items if there are any.
1. **Vector Search**\
   `Retriever.query()` embeds the user query (OpenAI or local) and fetches the top‑*k* FAQ questions whose cosine **similarity** ≥ `0.2` (can be configured).
2. **Prompt Assembly**\
   `Generator.prepare_messages()` injects those Q&A pairs plus recent chat history (last N turns) into an OpenAI system/user prompt.
3. **Generation**\
   `Generator.stream_completion()` streams tokens; server yields them instantly to the client.
4. **Conversation Memory**\
   After streaming, the assistant’s full reply is appended to `conversations[session_id]`, which is utilized to provide more contextualized responses.

---