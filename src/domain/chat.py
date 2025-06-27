from typing import Protocol

# Main application interfaces for the chatbot


class RetrieverPort(Protocol):
    def query(self, query: str) -> list[dict]: ...


class GeneratorPort(Protocol):
    async def stream_response(
        self, context: list[dict], user_query: str, chat_history: list[dict]
    ): ...
