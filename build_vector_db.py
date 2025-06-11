import pickle
from pathlib import Path
import argparse
import os
import openai

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from dotenv import load_dotenv
load_dotenv() # load environment variables from .env file

def build_vector_db(data_path: str, collection_name: str, use_local_embedding: bool, save_dir: str = "chroma_db") -> None:
    
    # read the data
    with open(data_path, "rb") as f:
        docs = pickle.load(f)

    # select the embedding method
    if use_local_embedding:
        embed_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in the .env file.")
        embed_f = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    
    # create db store
    client = chromadb.PersistentClient(path=save_dir)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_f)

    items = []
    for idx, q in enumerate(docs):
        question = str(q).strip()
        answer = str(docs[q]).strip()
        items.append({"id": str(idx), "question": question, "answer": answer})

    # filter out items in the collection if they already exist
    if collection.count() > 0:
        existing_ids = set(collection.get()["ids"])
        print(f"Collection '{collection_name}' already has {len(existing_ids)} items. Filtering out existing items...")
        items = [it for it in items if it["id"] not in existing_ids]
    if not items:
        print("No FAQ pairs to index. Exiting without indexing.")
        return

    total_items = len(items)
    print(f"Preparing to index {total_items} FAQ pairs into the collection...")

    if not use_local_embedding:
        # process the items in batches to avoid hitting OpenAI API limits
        batch_size = 100
        items = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        for batch in items:
            questions = [it["question"] for it in batch]
            answers = [it["answer"] for it in batch]
            collection.add(documents=questions, metadatas=[{"answer": ans} for ans in answers], ids=[it["id"] for it in batch])
    else:
        collection.add(documents=[it["question"] for it in items],
                    metadatas=[{"answer": it["answer"]} for it in items],
                    ids=[it["id"] for it in items])

    print(f"Finished indexing {total_items} FAQ pairs into the collection '{collection_name}'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the FAQ data file (pickle format).")
    parser.add_argument("--collection_name", type=str, default="SmartStore_FAQ", help="Name of the Chroma collection for the vector database.")
    parser.add_argument("--use_local_embedding", type=bool, default=False, 
                        help="OpenAI model (text-embedding-3-small) is used by default. Set to True to use local embedding model (SentenceTransformer: all-MiniLM-L6-v2).")
    parser.add_argument("--save_dir", type=str, default="chroma_db", help="Directory to save the Chroma vector database.")

    args = parser.parse_args()

    build_vector_db(args.data_dir, args.collection_name, args.use_local_embedding, args.save_dir)


