import asyncio
import json
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langchain_groq import ChatGroq
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from mistralai.client import Mistral

from RAGasEvaluator import RAGasEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "tests/evaluation"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from utils.config import (
    GROQ_API_KEY,
    MISTRAL_API_KEY,
    SEARCH_K,
)

from utils.vector_store import VectorStoreManager


# ============================================================
# Configuration
# ============================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# LLM GPT
# ============================================================

groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

ragas_llm = llm_factory(
    model="openai/gpt-oss-20b",
    provider="openai",
    client=groq_client,
    temperature=0.0,
    max_tokens=4096,
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=GROQ_API_KEY
    )


# ============================================================
# Mistral
# ============================================================

mistral_client = Mistral(
    api_key=MISTRAL_API_KEY
)


# ============================================================
# Vector store
# ============================================================

vector_store = VectorStoreManager()


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """Tu es un assistant expert sur la NBA.

Réponds à la question de l'utilisateur en utilisant uniquement
les informations présentes dans le contexte fourni.

Règles :
- Ne fabrique aucune information.
- Si le contexte ne permet pas de répondre, indique-le.
- Réponds de manière concise et précise.
"""


# ============================================================
# Génération de réponse
# ============================================================

async def generate_answer(
    question: str,
    contexts: list[str],
) -> str:

    context_str = "\n\n---\n\n".join(
        contexts
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=(
                f"CONTEXTE :\n\n"
                f"{context_str}\n\n"
                f"QUESTION :\n"
                f"{question}"
            )
        ),
    ]

    response = await llm.ainvoke(
        messages
    )

    return str(response.content)


# ============================================================
# Evaluation
# ============================================================

async def evaluate_all():

    # ---------------------------------
    # Chargement des cas de test
    # ---------------------------------

    with open(
        "tests/evaluation/rag_test_cases.json",
        "r",
        encoding="utf-8",
    ) as f:

        test_data = json.load(f)

    # ---------------------------------
    # Evaluateur Ragas
    # ---------------------------------

    async with Mistral(api_key=MISTRAL_API_KEY) as mistral_client:

        evaluator = RAGasEvaluator(
            vector_store=vector_store,
            mistral_client=mistral_client,
            llm=ragas_llm,
            search_k=SEARCH_K,
            answer_fn=generate_answer,
        )

        results = []

    # ---------------------------------
    # Evaluation de chaque question
    # ---------------------------------

        for i, item in enumerate(test_data):

            question = item["question"]
            reference = item["reference"]

            logging.info(
                "[%d/%d] %s",
                i + 1,
                len(test_data),
                question,
            )

            result = await evaluator.evaluate_case(
                question=question,
                reference=reference,
            )
            if result is None:
                logging.error(
                    "Résultat absent pour la question : %s",
                    question,
                )
                continue

            results.append(result)

            print()
            print("=" * 80)
            print(question)
            print("=" * 80)

            print(
                "Faithfulness :",
                result["faithfulness"],
            )

            print(
                "Response Relevancy :",
                result["response_relevancy"],
            )

            print(
                "Context Precision :",
                result["context_precision"],
            )

            print(
                "Context Recall :",
                result["context_recall"],
            )

        return results


# ============================================================
# Appel evaluate en asynchrone
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        evaluate_all()
    )