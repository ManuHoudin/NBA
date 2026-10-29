
from pathlib import Path
import sys
import time

# Ajouter la racine du projet au chemin Python.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from utils.config import MODEL_NAME, DASHSCOPE_API_KEY
from sql.sql_tool import create_sql_tool


def main():
    # Configuration Qwen via DashScope.
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        base_url=(
            "https://dashscope-intl.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        temperature=0,
        max_tokens=256,
        extra_body={
            "enable_thinking": False,
        },
    )

    # Initialiser l'outil une seule fois.
    ask_database = create_sql_tool(llm)

    question = input(
        "Pose une question sur la base NBA : "
    ).strip()

    if not question:
        print("Aucune question saisie.")
        return

    start = time.perf_counter()
    response = ask_database(question)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"QUESTION : {response['question']}")
    print(f"TEMPS TOTAL : {elapsed:.2f} secondes")

    if response["error"]:
        print("\nERREUR :")
        print(response["error"])
    else:
        print("\nREQUÊTE SQL GÉNÉRÉE :")
        print(response["query"])

        print("\nRÉSULTAT :")
        for row in response["result"]:
            print(row)

        print(
            f"\nNombre de lignes retournées : "
            f"{len(response['result'])}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
