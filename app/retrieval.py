from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def retrieve(
    self,
    query,
    top_k=5,
    max_distance=None,
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

    if max_distance is not None:
        results = results[
            results["distance"] <= max_distance
        ]

    return results