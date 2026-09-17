import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


class Generator:
    def __init__(self, model="openai/gpt-oss-120b"):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN environment variable is not set."
            )

        self.client = InferenceClient(
            token=token
        )

        self.model = model

    def generate(self, question, results):
        if len(results) == 0:
            return (
                "The retrieved research does not contain "
                "enough information to answer this."
            )

        context = "\n\n".join(
            f"Abstract ID: {row.abstract_id}\n"
            f"{row.abstract_text}"
            for _, row in results.iterrows()
        )

        prompt = f"""
You are an AI biomedical research assistant.

Answer the question ONLY using the supplied research abstracts.

Rules:
- Give a concise answer in 1-2 complete sentences.
- Do not speculate.
- Do not add information that is not supported by the abstracts.
- Do not provide citations or source IDs.
- If the abstracts do not contain enough information, say:
  "The retrieved research does not contain enough information to answer this."

RESEARCH:

{context}

QUESTION:

{question}

ANSWER:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()