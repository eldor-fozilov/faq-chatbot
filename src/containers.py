from __future__ import annotations

from dependency_injector import containers, providers

from src.adapters.retriever import Retriever
from src.adapters.generator import Generator
from src.utils import parse_bool


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    retriever = providers.Singleton(
        Retriever,
        collection_name=config.collection_name,
        use_local_embedding=config.use_local_embedding,
        data_path=config.data_path,
        save_dir=config.db_dir,
        top_k=config.top_k,
        sim_threshold=config.sim_threshold,
    )

    generator = providers.Singleton(
        Generator,
        model_name=config.model_name,
        num_turns=config.num_turns,
        api_key=config.api_key,
    )

    # set config
    # main
    config.api_key.from_env("OPENAI_API_KEY", None)
    if not config.api_key():
        raise ValueError(
            "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in the .env file."
        )
    config.use_local_embedding.from_env("USE_LOCAL_EMBEDDING", "false", as_=parse_bool)
    config.collection_name.from_env("COLLECTION_NAME", "SmartStore_FAQ")
    config.model_name.from_env("OPENAI_MODEL", "gpt-4o-mini")
    config.db_dir.from_env("DB_DIR", "chroma_db")
    # additional
    config.data_path.from_env("DATA_PATH", "data/final_result.pkl")
    config.top_k.from_env("TOP_K", 4, as_=int)
    config.num_turns.from_env("NUM_TURNS", 8, as_=int)
    config.sim_threshold.from_env("SIM_THRESHOLD", 0.2, as_=float)
