import pickle
import argparse
import os
import openai

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from dotenv import load_dotenv
load_dotenv() # load environment variables from .env file


class Retriever:
    def __init__(self, collection_name: str, use_local_embedding: bool, data_path: str,
                  save_dir: str = "chroma_db", top_k: int = 4, sim_threshold: float = 0.3):

        self.collection_name = collection_name
        self.use_local_embedding = use_local_embedding
        self.data_path = data_path
        self.save_dir = save_dir
        self.top_k = top_k
        self.sim_threshold = sim_threshold

        # select the embedding method
        if use_local_embedding:
            self.embed_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in the .env file.")
            self.embed_f = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=openai.api_key)

        # create db store
        self.client = chromadb.PersistentClient(path=save_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embed_f)

    def build_vector_db(self):

        # read the data
        with open(self.data_path, "rb") as f:
            docs = pickle.load(f)
        
        print(f"Loaded {len(docs)} FAQ pairs from {self.data_path}.")

        # select the embedding method
        if self.use_local_embedding:
            embed_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2",
                                                                               metadata={"hnsw:space": "cosine"})
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in the .env file.")
            embed_f = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        
        # create db store
        client = chromadb.PersistentClient(path=self.save_dir)
        collection = client.get_or_create_collection(name=self.collection_name, embedding_function=embed_f,
                                                     metadata={"hnsw:space": "cosine"})

        items = []
        for idx, q in enumerate(docs):
            question = str(q).strip()
            answer = str(docs[q]).strip()
            items.append({"id": str(idx), "question": question, "answer": answer})

        # filter out items in the collection if they already exist
        if collection.count() > 0:
            existing_ids = set(collection.get()["ids"])
            print(f"Collection '{self.collection_name}' already has {len(existing_ids)} items. Filtering out existing items...")
            items = [it for it in items if it["id"] not in existing_ids]
        if not items:
            print("No FAQ pairs to index. Exiting without indexing.")
            return

        total_items = len(items)

        batch_size = 100
        items = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # show progress
        print(f"Indexing {total_items} FAQ pairs into the collection '{self.collection_name}' in batches of {batch_size}...")
        from tqdm import tqdm
        for batch in tqdm(items, desc=f"Indexing FAQ pairs.", unit="batch"):
            questions = [it["question"] for it in batch]
            answers = [it["answer"] for it in batch]
            collection.add(documents=questions, metadatas=[{"answer": ans} for ans in answers], ids=[it["id"] for it in batch])

        print(f"Finished indexing {total_items} FAQ pairs into the collection '{self.collection_name}'.")

    def query(self, query: str, top_k: int = None, sim_threshold: float = None):

        if top_k is None:
            top_k = self.top_k
        if sim_threshold is None:
            sim_threshold = self.sim_threshold

        results = self.collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])

        docs, metas, dists = results["documents"], results["metadatas"], results["distances"]
        rows = []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            # convert cosine distance to cosine similarity
            sim = 1 - dist
            if sim < sim_threshold:
                continue
            rows.append({"question": doc, "answer": meta["answer"], "dist": dist})

        return rows

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.getenv("DATA_PATH", "data/final_result.pkl"), help="Path to the FAQ data file (pickle format).")
    parser.add_argument("--collection_name", type=str, default=os.getenv("COLLECTION_NAME", "SmartStore_FAQ"), help="Name of the Chroma collection for the vector database.")
    parser.add_argument("--use_local_embedding", type=bool, default=os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true",
                        help="OpenAI model (text-embedding-3-small) is used by default. Set to True to use local embedding model (SentenceTransformer: all-MiniLM-L6-v2).")
    parser.add_argument("--top_k", type=int, default=4, help="Number of top results to return from the vector database.")
    parser.add_argument ("--build_db", action='store_true', help="Build the vector database from the FAQ data file.")
    parser.add_argument("--query_db", action='store_true', help="Query the vector database with a user-defined query.")
    parser.add_argument("--query", type=str, default="스마트스토어에 입점하려면 어떻게 해야 하나요?", help="User-defined query to test the vector database.")
    parser.add_argument("--save_dir", type=str, default=os.getenv("DB_DIR", "chroma_db"), help="Directory to save the Chroma vector database.")
    args = parser.parse_args()

    retriever = Retriever(collection_name=args.collection_name, use_local_embedding=args.use_local_embedding, data_path=args.data_path, save_dir=args.save_dir)
    # build the vector database
    if args.build_db:
        retriever.build_vector_db()
    # example query
    if args.query_db:
        if not args.query:
            raise ValueError("Please provide a query using --query argument.")
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file {args.data_path} does not exist. Please provide a valid path to the FAQ data file.")
        if not retriever.collection.count():
            print(f"Collection '{args.collection_name}' is empty. Please build the vector database first using --build_db.")
            exit(1)
        
        results = retriever.query(args.query)
        print(f"Query: {args.query}")
        for idx, res in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"  Question: {res['question']}")
            print(f"  Answer: {res['answer']}")
            print(f"  Distance: {res['dist']:.4f}")

