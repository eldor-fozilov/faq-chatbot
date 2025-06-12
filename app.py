import os
from retriever import Retriever
from generator import Generator

import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")  # expect key in env or .env

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "SmartStore_FAQ")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA_PATH = os.getenv("DATA_PATH", "data/final_result.pkl")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"

TOP_K = 8 # number of top results to return from the vector DB

retriever = Retriever(
    collection_name=COLLECTION_NAME,
    use_local_embedding=USE_LOCAL_EMBEDDING,
    data_dir=DATA_PATH,
    save_dir=DB_DIR,
    top_k=TOP_K
)
retriever.build_vector_db() # build the vector DB if not already done

generator = Generator(model_name=MODEL_NAME)

app = FastAPI(title="SmartStore FAQ Chatbot")

# store conversations in memory
conversations = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    sess_id = req.session_id
    user_query = req.message.strip()

    chat_hist = conversations.setdefault(sess_id, [])

    context = retriever.query(user_query)
    
    messages, oos_reply = generator.prepare_messages(context, user_query, chat_hist)

    # save user turn
    chat_hist.append({"role": "user", "content": user_query})

    if oos_reply is not None:
        chat_hist.append({"role": "assistant", "content": oos_reply})
        return StreamingResponse(
            iter([oos_reply + "\n\n"]),
            media_type="text/plain"
        )

    async def streamer():
        assistant_buf = []  # collect chunks for history

        async for chunk in generator.stream_completion(messages):
            assistant_buf.append(chunk)
            yield chunk
        
        yield "\n\n"

        # save assistant turn
        assistant_reply = "".join(assistant_buf).strip()
        chat_hist.append({"role": "assistant", "content": assistant_reply})

    return StreamingResponse(streamer(), media_type="text/plain")

# ---- run check ---- #
@app.get("/")
async def root():
    return {"status": "ok"}