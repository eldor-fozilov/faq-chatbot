
from openai import AsyncOpenAI
import json
import os
from utils import OOS_PROMPT, SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

class Generator:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2, num_turns: int = 8):
        self.model_name = model_name

        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        self.temperature = temperature
        self.num_turns = num_turns # number of turns to include in the chat history

    def prepare_messages(self, context: list, user_query: str, chat_history: list, ans_in_chat_hist: bool = True):

        if not context:
            return None, OOS_PROMPT

        context_block = "\n".join(
            f"질문: {r['question']}\n답변: {r['answer']}" for r in context
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"관련 FAQ:\n{context_block}"},
        ]

        if ans_in_chat_hist:
            # include both user and assistant turns in the history
            for turn in chat_history[-self.num_turns:]:
                messages.append(turn)
        else:
            # only include user turns in the history
            messages.extend(
                {"role": "user", "content": turn["content"]}
                for turn in chat_history[-self.num_turns:] if turn["role"] == "user"
            )
        
        messages.append({"role": "user", "content": user_query})

        return messages, None

    async def stream_completion(self, messages: list):
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            yield delta if delta else ""


if __name__ == "__main__":
    import asyncio
    generator = Generator()

    demo_context = [
        {
            "question": "스마트스토어에 어떻게 등록하나요?",
            "answer": "스마트스토어에 등록하려면 네이버 판매자센터에 접속해 절차를 따라야 합니다.",
        },
        {
            "question": "스마트스토어에서 판매 가능한 상품은 무엇인가요?",
            "answer": "대부분의 상품을 판매할 수 있으나 일부 카테고리는 규제를 받습니다.",
        },
    ]
    user_query = "스마트스토어에 등록하려면 어떻게 해야 하나요?"
    history = []

    messages, oos = generator.prepare_messages(demo_context, user_query, history)

    async def main():
        if oos:
            print(oos)
            return

        print("\n=== streaming ===\n")
        async for chunk in generator.stream_completion(messages):
            print(chunk, end="", flush=True)
        print("\n\n=== streaming complete ===\n")

    asyncio.run(main())