from openai import AsyncOpenAI
from src.domain.chat import GeneratorPort
from src.utils import OOS_PROMPT, SYSTEM_PROMPT, QUERY_REFINEMENT_PROMPT
from pydantic import BaseModel, Field


class QueryFormat(BaseModel):
    refine_query: bool = Field(
        ..., description="질문을 실제로 수정했으면 true, 수정이 필요 없으면 false."
    )
    refined_query: str = Field(..., description="최종 한국어 질문 문장.")


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
            # include both user and assistant turns from the chat history
            for turn in chat_history[-self.num_turns :]:
                messages.append(turn)
        else:
            # only include user turns from the chat history
            messages.extend(
                {"role": "user", "content": turn["content"]}
                for turn in chat_history[-self.num_turns :]
                if turn["role"] == "user"
            )

        messages.append({"role": "user", "content": user_query})

        return messages

    async def refine_query(self, user_query: str, chat_history: list[dict]):
        messages = [
            {"role": "system", "content": QUERY_REFINEMENT_PROMPT},
        ]
        for turn in chat_history[-self.num_turns :]:
            messages.append(turn)

        messages.append(
            {"role": "user", "content": "변환할 사용자 질문: " + user_query}
        )

        try:
            resp = await self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=QueryFormat,
            )
        except Exception as e:
            print(
                f"Error during query refinement: {e}"
            )  #  refusal, content_filter, etc.
            return None

        resp = resp.choices[0].message

        if not resp.parsed.refine_query:
            return None

        refined_query = resp.parsed.refined_query.strip()
        return refined_query
