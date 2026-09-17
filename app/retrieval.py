from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        index_path,
        documents_path,
        model_path=None,
    ):
        # Load the local embedding model
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "all-MiniLM-L6-v2"
            )

        self.embedding_model = SentenceTransformer(str(model_path))

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load documents
        self.documents = pd.read_pickle(documents_path)

    def retrieve(
        self,
        query,
        top_k=5,
        max_distance=0.65,
    ):
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = self.documents.iloc[indices[0]].copy()

        results["distance"] = distances[0]

        # Remove weak matches
        results = results[
            results["distance"] <= max_distance
        ]

        return results