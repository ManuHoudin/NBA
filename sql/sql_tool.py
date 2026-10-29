import re
import sqlite3
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_classic.chains import create_sql_query_chain


# Chemin de la base SQLite
DB_PATH = Path(__file__).resolve().parent / "nba.db"


def create_sql_tool(llm, db_path=DB_PATH):
    """
        Crée un outil qui :
    1. génère une requête SQL avec LangChain ;
    2. vérifie qu'il s'agit d'une requête de lecture ;
    3. exécute la requête en lecture seule ;
    4. retourne la requête et son résultat.

    Un seul appel au LLM est effectué par question.
    """

    db_path = Path(db_path).resolve()

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Base SQLite introuvable : {db_path}"
        )

    # Connexion utilisée par LangChain pour découvrir le schéma.
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    # La chaîne LangChain génère la requête à partir de la question.
    query_chain = create_sql_query_chain(
        llm=llm,
        db=db,
        k=10,
    )

    def ask_database(question: str) -> dict:
        """Interroge la base avec un seul appel au LLM."""

        try:
            raw_query = query_chain.invoke({
                "question": question
            })

            # Nettoyer la réponse.
            query = raw_query.strip()

            query = re.sub(
                r"^```(?:sql)?\s*|\s*```$",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()

            query = re.sub(
                r"^SQLQuery:\s*",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()

            # Si le modèle ajoute un préambule ou du texte, extraire le code SQL.
            match = re.search(
                r"\b(SELECT|WITH)\b[\s\S]*",
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                raise ValueError(
                    f"Aucune requête SELECT détectée : {raw_query!r}"
                )

            query = match.group(0).strip()

            # Autoriser un point-virgule final, mais pas plusieurs
            # instructions SQL.
            query = query.rstrip().removesuffix(";").strip()

            if ";" in query:
                raise ValueError(
                    "Plusieurs instructions SQL sont interdites."
                )

            # Exécuter la requête avec SQLite en lecture seule.
            # Le préfixe file: et mode=ro empêchent les écritures.
            sqlite_uri = (
                f"file:{db_path.as_posix()}?mode=ro"
            )

            with sqlite3.connect(
                sqlite_uri,
                uri=True,
                timeout=10,
            ) as conn:
                conn.row_factory = sqlite3.Row

                # Garde-fou SQLite supplémentaire.
                conn.set_authorizer(
                    lambda action, arg1, arg2, db_name, trigger: (
                        sqlite3.SQLITE_OK
                        if action in (
                            sqlite3.SQLITE_SELECT,
                            sqlite3.SQLITE_READ,
                            sqlite3.SQLITE_FUNCTION,
                        )
                        else sqlite3.SQLITE_DENY
                    )
                )

                cursor = conn.execute(query)
                rows = [dict(row) for row in cursor.fetchall()]

            return {
                "question": question,
                "query": query,
                "result": rows,
                "error": None,
            }

        except Exception as exc:
            return {
                "question": question,
                "query": None,
                "result": None,
                "error": str(exc),
            }

    return ask_database
