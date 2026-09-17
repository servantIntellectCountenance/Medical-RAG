from .retrieval import Retriever
from .generation import Generator


class MedicalRAG:
    def __init__(
        self,
        index_path,
        documents_path,
        model_path=None,
    ):
        self.retriever = Retriever(
            index_path=index_path,
            documents_path=documents_path,
            model_path=model_path,
        )

        self.generator = Generator()

    def ask(self, question, top_k=5):
        # Retrieve relevant research
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # Generate answer
        answer = self.generator.generate(
            question=question,
            results=results,
        )

        # Extract source IDs
        source_ids = [
            str(row.abstract_id)
            for _, row in results.iterrows()
        ]

        return {
            "answer": answer,
            "sources": source_ids,
            "results": results,
        }