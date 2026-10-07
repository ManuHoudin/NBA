import json
import logging

from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from mistralai.client import MistralClient
from openai import AsyncOpenAI


from utils.config import DASHSCOPE_API_KEY, MODEL_NAME, SEARCH_K, MISTRAL_API_KEY
from utils.vector_store import VectorStoreManager



load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------
# Configuration Qwen
# ---------------------------------------------------------

ragas_client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

ragas_llm = llm_factory(
    model=MODEL_NAME,
    client=ragas_client,
    temperature=0.2,
    extra_body={
        "enable_thinking": False
    }
)

# ---------------------------------------------------------
# Formatage embeddings Mistral pour Ragas
# ---------------------------------------------------------
from ragas.embeddings.base import BaseRagasEmbeddings
from mistralai.client import MistralClient


class MistralRagasEmbeddings(BaseRagasEmbeddings):

    def __init__(self, mistral_client):
        self.mistral_client = mistral_client

    def embed_query(self, text: str) -> list[float]:
        response = self.mistral_client.embeddings(
            model="mistral-embed",
            input=[text],
        )

        if not response.data:
            raise RuntimeError("Mistral n'a retourné aucun embedding")

        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.mistral_client.embeddings(
            model="mistral-embed",
            input=texts,
        )

        if not response.data:
            raise RuntimeError("Mistral n'a retourné aucun embedding")

        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.embed_documents(texts)

# ---------------------------------------------------------
# Configuration Mistral
# ---------------------------------------------------------
mistral_client = AsyncOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
)

ragas_embeddings = MistralRagasEmbeddings(mistral_client=mistral_client)

# ---------------------------------------------------------
# Métriques d'évaluation
# ---------------------------------------------------------

metrics=[
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings,),
]

# ---------------------------------------------------------
# Prompt RAG
# ---------------------------------------------------------

SYSTEM_PROMPT = """Tu es un assistant expert sur la NBA.

Réponds à la question de l'utilisateur en utilisant uniquement
les informations présentes dans le contexte fourni.

Règles :
- Ne fabrique aucune information.
- Si le contexte ne permet pas de répondre, indique-le.
- Réponds de manière concise et précise.
"""


# ---------------------------------------------------------
# Génération de la réponse
# ---------------------------------------------------------

def generate_answer(question, contexts):

    context_str = "\n\n---\n\n".join(contexts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"CONTEXTE :\n\n"
                f"{context_str}\n\n"
                f"QUESTION :\n"
                f"{question}"
            )
        )
    ]

    response = ragas_llm.invoke(messages)

    return response.content


# ---------------------------------------------------------
# Chargement du dataset
# ---------------------------------------------------------

with open(
    "tests/evaluation/rag_test_cases.json",
    "r",
    encoding="utf-8"
) as f:
    test_data = json.load(f)


# ---------------------------------------------------------
# Initialisation du Vector Store
# ---------------------------------------------------------

vector_store = VectorStoreManager()


# ---------------------------------------------------------
# Construction du dataset Ragas
# ---------------------------------------------------------

ragas_data = []


for i, item in enumerate(test_data):

    question = item["question"]
    reference = item["reference"]

    logging.info(
        f"[{i + 1}/{len(test_data)}] {question}"
    )

    # Recherche FAISS
    search_results = vector_store.search(
        question,
        k=SEARCH_K
    )

    contexts = [
        result["text"]
        for result in search_results
    ]

    # Génération Qwen
    answer = generate_answer(
        question,
        contexts
    )

    ragas_data.append({
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": reference
    })


# ---------------------------------------------------------
# Dataset Ragas
# ---------------------------------------------------------

dataset = Dataset.from_list(ragas_data)


# ---------------------------------------------------------
# Evaluation Ragas
# ---------------------------------------------------------

result = evaluate(
    dataset,
    metrics=metrics
    
)


print("\n==============================")
print("RESULTATS RAGAS")
print("==============================")

print(result)