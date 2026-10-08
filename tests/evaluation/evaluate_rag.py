import asyncio
import json
import logging

from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI

from mistralai.client import Mistral

from tests.evaluation.RAGasEvaluator import RAGasEvaluator

from utils.config import (
    DASHSCOPE_API_KEY,
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
# LLM Qwen
# ============================================================

llm = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=DASHSCOPE_API_KEY,
    base_url=(
        "https://dashscope-intl.aliyuncs.com/"
        "compatible-mode/v1"
    ),
    temperature=0.2,
    extra_body={
        "enable_thinking": False
    },
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

    # NOUVEAU : On ouvre proprement le client Mistral en mode Asynchrone
    # pour que l'évaluateur Ragas et vos embeddings puissent l'utiliser sans planter
    async with Mistral(api_key=MISTRAL_API_KEY) as mistral_client:

        evaluator = RAGasEvaluator(
            vector_store=vector_store,
            mistral_client=mistral_client,
            llm=llm,
            search_k=SEARCH_K,
            answer_fn=generate_answer,
        )

        results = []

    # ---------------------------------
    # Evaluation de chaque question
    # ---------------------------------

        for i, item in enumerate(test_data):

            question = item["question"]

            logging.info(
                "[%d/%d] %s",
                i + 1,
                len(test_data),
                question,
            )

            result = await evaluator.evaluate_case(
                question=question,
                reference=item["reference"],
            )

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

        return results


# ============================================================
# Appel evaluate en asynchrone
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        evaluate_all()
    )