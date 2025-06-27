from src.containers import Container

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

# DI Container initialization
container = Container()
container.init_resources()
retriever = container.retriever()
generator = container.generator()

# FastAPI app initialization
app = FastAPI(title="SmartStore FAQ Chatbot")

# store conversations in memory
conversations = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    sess_id, user_query = req.session_id, req.message.strip()
    chat_hist = conversations.setdefault(sess_id, [])

    if chat_hist:
        refined_query = await generator.refine_query(
            user_query, chat_hist
        )  # refine query if needed (based on chat history)

        if refined_query is not None:
            # print(f"Original query: {user_query}")
            # print(f"Refined query: {refined_query.strip()}")
            user_query = refined_query.strip()

    context = retriever.query(user_query)
    chat_hist.append({"role": "user", "content": user_query})

    async def streamer():
        assistant_buf = []  # collect chunks for history

        async for chunk in generator.stream_response(context, user_query, chat_hist):
            assistant_buf.append(chunk)
            yield chunk
        yield "\n\n"

        # save assistant turn
        assistant_reply = "".join(assistant_buf).strip()
        chat_hist.append({"role": "assistant", "content": assistant_reply})

    conversations[sess_id] = chat_hist

    return StreamingResponse(streamer(), media_type="text/plain")


# check run status
@app.get("/")
def root():
    return {"status": "ok"}


# reset conversations
@app.post("/reset")
def reset_conversations():
    global conversations
    conversations = {}
    return {"status": "conversations reset"}
