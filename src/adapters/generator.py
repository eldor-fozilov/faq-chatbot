from openai import AsyncOpenAI
from src.domain.chat import GeneratorPort
from src.utils import OOS_PROMPT, SYSTEM_PROMPT


class Generator(GeneratorPort):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.2,
        num_turns: int = 8,
        ans_in_chat_hist: bool = True,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.ans_in_chat_hist = (
            ans_in_chat_hist  # whether to include assistant replies in chat history
        )
        self.num_turns = num_turns  # number of turns to include in the chat history

    # main interface method
    async def stream_response(
        self, context: list[dict], user_query: str, chat_history: list[dict]
    ):
        if not context:
            # if no context is provided, return out-of-scope response
            yield OOS_PROMPT
            return

        messages = self.prepare_messages(context, user_query, chat_history)

        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            yield delta if delta else ""

    def prepare_messages(
        self, context: list[dict], user_query: str, chat_history: list[dict]
    ) -> list[dict]:
        context_block = "\n".join(
            f"질문: {r['question']}\n답변: {r['answer']}" for r in context
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"관련 FAQ:\n{context_block}"},
        ]

        if self.ans_in_chat_hist:
            # include both user and assistant turns in the history
            for turn in chat_history[-self.num_turns :]:
                messages.append(turn)
        else:
            # only include user turns in the history
            messages.extend(
                {"role": "user", "content": turn["content"]}
                for turn in chat_history[-self.num_turns :]
                if turn["role"] == "user"
            )

        messages.append({"role": "user", "content": user_query})

        return messages
