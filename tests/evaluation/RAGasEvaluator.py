from __future__ import annotations

from ragas import SingleTurnSample

from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
)

from mistralai.client import Mistral


class MistralRagasEmbeddings(BaseRagasEmbeddings):

    def __init__(self, client: Mistral):
        self.client = client

    # -------------------------
    # Version synchrone
    # -------------------------

    def embed_query(self, text: str) -> list[float]:

        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[text],
        )

        embedding = response.data[0].embedding

        if embedding is None:
            raise RuntimeError(
                "Mistral n'a pas retourné d'embedding."
            )

        return embedding

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )

        embeddings = []

        for item in sorted(
            response.data,
            key=lambda item: (
                item.index
                if item.index is not None
                else -1
            ),
        ):
            if item.embedding is None:
                raise RuntimeError(
                    "Mistral n'a pas retourné d'embedding."
                )

            embeddings.append(item.embedding)

        return embeddings

    # -------------------------
    # Version asynchrone
    # -------------------------

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:

        response = await self.client.embeddings.create_async(
            model="mistral-embed",
            inputs=[text],
        )

        embedding = response.data[0].embedding

        if embedding is None:
            raise RuntimeError(
                "Mistral n'a pas retourné d'embedding."
            )

        return embedding

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        response = await self.client.embeddings.create_async(
            model="mistral-embed",
            inputs=texts,
        )

        embeddings = []

        for item in sorted(
            response.data,
            key=lambda item: (
                item.index
                if item.index is not None
                else -1
            ),
        ):
            if item.embedding is None:
                raise RuntimeError(
                    "Mistral n'a pas retourné d'embedding."
                )

            embeddings.append(item.embedding)

        return embeddings


class RAGasEvaluator:

    def __init__(
        self,
        vector_store,
        mistral_client,
        llm,
        search_k: int,
        answer_fn,
    ):

        self.vector_store = vector_store
        self.mistral_client = mistral_client
        self.llm = llm
        self.search_k = search_k
        self.answer_fn = answer_fn

        # Embeddings utilisés par ResponseRelevancy
        self.ragas_embeddings = MistralRagasEmbeddings(
            mistral_client
        )

        # Generation metrics
        self.faithfulness = Faithfulness(
            llm=llm
        )

        self.response_relevancy = ResponseRelevancy(
            llm=llm,
            embeddings=self.ragas_embeddings,
        )

    async def evaluate_case(
        self,
        question: str,
        reference: list[str | int],
    ) -> dict:

        print(">>> EVALUATE_CASE START")

        # ---------------------------------
        # Retrieval
        # ---------------------------------

        search_results = self.vector_store.search(
            question,
            k=self.search_k,
        )

        contexts = [
            result["text"]
            for result in search_results
        ]

        # ---------------------------------
        # Génération de la réponse
        # ---------------------------------

        answer = await self.answer_fn(
            question,
            contexts,
        )

        print("Answer :", answer)

        # ---------------------------------
        # Sample pour les métriques
        # ---------------------------------

        sample_rag = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        # ---------------------------------
        # Faithfulness
        # ---------------------------------

        faithfulness = await self.faithfulness.single_turn_ascore(
            sample_rag
        )

        print("Faithfulness :", faithfulness)

        # ---------------------------------
        # Response relevancy
        # ---------------------------------

        response_relevancy = (
            await self.response_relevancy.single_turn_ascore(
                sample_rag
            )
        )

        print(
            "Response Relevancy :",
            response_relevancy,
        )

        return {
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
        }