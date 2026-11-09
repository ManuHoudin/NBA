from __future__ import annotations

from ragas import SingleTurnSample

from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from mistralai.client import Mistral

import asyncio
from typing import Any
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from MistralChat import traiter_question_avec_tools


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
        self.ragas_embeddings = MistralRagasEmbeddings(mistral_client)

        # Generation metrics
        self.faithfulness = Faithfulness(llm=llm)
        self.response_relevancy = ResponseRelevancy(llm=llm,embeddings=self.ragas_embeddings,)

        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)

    async def answer_with_tools(self, question: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            traiter_question_avec_tools,
            question
    )

    async def evaluate_case(
        self,
        question: str,
        reference: str
    ) -> dict:

        print("\n>>> EVALUATE_CASE START", flush=True)

        # ---------------------------------
        # Génération de la réponse avec outils
        # ---------------------------------

        result = await self.answer_with_tools(question)
        if result.get("source") == "error":

            logging.error(
                "Evaluation RAGAS annulée : erreur de routage."
            )

            return {
                "question": question,
                "answer": result["answer"],
                "faithfulness": None,
                "response_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            }

        answer = result["answer"]
        actual_contexts = result.get("contexts", [])

        # Sécurisation des contextes
        if actual_contexts is None:
            actual_contexts = []

        actual_contexts = [
            str(context)
            for context in actual_contexts
            if context is not None
        ]

        print("\n--- DONNÉES POUR RAGAS ---", flush=True)
        print("Question :", question, flush=True)
        print("Source :", result.get("source"), flush=True)
        print("Answer :", answer, flush=True)
        print("Nombre de contextes :", len(actual_contexts), flush=True)

        for i, context in enumerate(actual_contexts):
            print(f"Contexte {i + 1} :", context, flush=True)

        # ---------------------------------
        # Sample RAGAS
        # ---------------------------------

        sample_rag = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=reference,
            retrieved_contexts=actual_contexts,
        )

        # ---------------------------------
        # Fonction d'évaluation sécurisée
        # ---------------------------------

        async def evaluate_metric(name, metric):

            print(f"\n>>> Calcul de {name}...", flush=True)

            try:
                score = await metric.single_turn_ascore(sample_rag)

                print(f"{name} : {score}", flush=True)

                return score

            except Exception:

                logging.exception(
                    "Erreur pendant le calcul de %s",
                    name
                )

                return None

        # ---------------------------------
        # Calcul des métriques
        # ---------------------------------

        faithfulness = await evaluate_metric(
            "Faithfulness",
            self.faithfulness
        )

        response_relevancy = await evaluate_metric(
            "Response Relevancy",
            self.response_relevancy
        )

        context_precision = await evaluate_metric(
            "Context Precision",
            self.context_precision
        )

        context_recall = await evaluate_metric(
            "Context Recall",
            self.context_recall
        )

        # ---------------------------------
        # Résultats
        # ---------------------------------

        return {
            "question": question,
            "answer": answer,
            "contexts": actual_contexts,
            "source": result.get("source"),
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }