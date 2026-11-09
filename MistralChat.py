import streamlit as st
import logging
import logfire
import json
import time
import re


from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings
from typing import Any, Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from sql.sql_tool import create_sql_tool

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME, GROQ_API_KEY
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# Configure l'observabilité
logfire.configure()

# Trace les exécutions de l'agent Pydantic AI
logfire.instrument_pydantic_ai()

# Trace les validations Pydantic qui échouent
logfire.instrument_pydantic(record="failure")

# Les modèles Pydantic sont importés ensuite
from utils.format_control import UserQuery, RAGResponse, SearchResult

# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral / GPT ---
api_key = GROQ_API_KEY
model = MODEL_NAME

# --- Initialisation du modèle GPT ---
groq_llm = ChatGroq(
    model=model,
    temperature=0.2,
    api_key=api_key
    )


if not api_key:
    st.error("Erreur : Clé API GPT non trouvée (GROQ_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()


# Initialisation du modèle Groq pour Pydantic AI
groq_model = GroqModel(
    model_name=model,
    provider=GroqProvider(api_key=api_key)
)

# Prompt pour l'agent Pydantic AI
rag_agent = Agent(
    groq_model,
    output_type=RAGResponse,
    model_settings=GroqModelSettings(groq_reasoning_format="hidden", groq_reasoning_effort="low"),
    system_prompt=(
        "Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA. "
        "Tu réponds aux questions des fans en te basant uniquement sur le contexte "
        "fourni par la base de connaissances. "
        "Si le contexte ne permet pas de répondre, indique-le clairement."
    ),
)

# --- Configuration SQL ---
ask_database = create_sql_tool(groq_llm)

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]


# --- Outils LangChain : SQL et FAISS ---

def build_nba_tools(ask_database, vector_store_manager):

    @tool("nba_sql_tool")
    def nba_sql_tool(question: str) -> str:
        """Interroge les tables SQL NBA pour les questions chiffrées :
        statistiques de joueurs, classements, moyennes, totaux,
        comparaisons statistiques et statistiques par équipe."""
        response = ask_database(question)

        if response["error"]:
            return json.dumps(
                {"error": response["error"]},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "query": response["query"],
                "result": response["result"],
            },
            ensure_ascii=False,
            default=str,
        )

    @tool("faiss_search_tool")
    def faiss_search_tool(question: str) -> str:
        """Recherche dans la base documentaire FAISS les informations
        utiles aux questions qualitatives, explicatives ou contextuelles
        sur la NBA. Retourne les passages pertinents trouvés."""
        if vector_store_manager is None:
            return "Erreur : la base documentaire FAISS est indisponible."

        print("\n--- FAISS CALLED ---")
        print("Query:", question)

        try:
            search_results = [
                SearchResult(**result)
                for result in vector_store_manager.search(
                    question,
                    k=SEARCH_K,
                )
            ]

            if not search_results:
                return "Aucun passage pertinent trouvé dans FAISS."

            return "\n\n---\n\n".join(
                f"Source : {res.metadata.get('source', 'Inconnue')}\n"
                f"Score : {res.score}\n"
                f"Contenu : {res.text}"
                for res in search_results
            )

        except Exception as exc:
            logging.exception("Erreur pendant la recherche FAISS")
            return f"Erreur pendant la recherche documentaire : {exc}"

    return [nba_sql_tool, faiss_search_tool]


nba_tools = build_nba_tools(
    ask_database=ask_database,
    vector_store_manager=vector_store_manager,
)

nba_tools_by_name = {tool.name: tool for tool in nba_tools}

# Le modèle appelle les outils.
groq_with_tools = groq_llm.bind_tools(nba_tools)


class ResearchQuery(BaseModel):
    tool: Literal["nba_sql_tool", "faiss_search_tool"]

    objective: str = Field(
        description="Information précise que cette requête doit obtenir."
    )

    question: str = Field(
        description="Question en langage naturel à transmettre à l'outil."
    )


class ResearchPlan(BaseModel):
    queries: list[ResearchQuery] = Field(
        description="Liste complète des recherches nécessaires."
    )

# --- Routage et synthèse avec LangChain ---

ROUTER_SYSTEM_PROMPT = """
Tu es un planificateur de recherches pour un assistant spécialisé NBA.

Tu disposes de deux outils :

1. nba_sql_tool
Pour les statistiques structurées :
- statistiques individuelles et collectives ;
- points, rebonds, passes ;
- moyennes, totaux et pourcentages ;
- comparaisons et classements ;
- statistiques par saison ou par match.

2. faiss_search_tool
Pour les informations documentaires :
- biographies ;
- comptes rendus de matchs ;
- actualités ;
- analyses tactiques ;
- opinions et réactions des fans.

MISSION :

Analyse la question et prépare la liste complète des recherches
nécessaires pour construire une réponse fiable.

RÈGLES DE PLANIFICATION :

1. Identifie toutes les informations distinctes nécessaires.

2. Regroupe les statistiques dans une seule requête SQL
   lorsqu'elles peuvent être obtenues ensemble.

3. Autorise plusieurs requêtes SQL lorsqu'elles répondent
   à des objectifs réellement différents.

4. Autorise plusieurs recherches FAISS si elles sont nécessaires.

5. Utilise les deux outils lorsque la question nécessite
   de croiser des statistiques et des informations documentaires.

6. Ne crée jamais plusieurs requêtes pour reformuler,
   vérifier ou confirmer une même information.

7. Chaque requête doit correspondre à un objectif distinct.

8. Ne prévois jamais une requête destinée à remplacer
   ou vérifier le résultat d'une autre requête.

9. N'invente aucune statistique, table, colonne ou champ.

10. Les questions destinées à SQL doivent être formulées
    en langage naturel, jamais en SQL.

11. Respecte les noms et la signification des statistiques
    et colonnes de la base.

12. N'utilise pas FAISS pour rechercher des statistiques
    déjà disponibles dans SQL.

13. Ne prévois pas de recherche lorsque l'information
    n'est pas nécessaire pour répondre à la question.

IMPORTANT :

Une première requête qui répond directement à une partie
de la question ne dispense pas de rechercher les autres
informations distinctes nécessaires.

En revanche, une information déjà couverte par une requête
ne doit jamais faire l'objet d'une seconde recherche.

Prépare un plan complet avant toute exécution.
"""


def traiter_question_avec_tools(question: str) -> dict[str, Any]:
    """
    Architecture :
    1. Planification des recherches par le LLM.
    2. Validation et déduplication du plan.
    3. Exécution des requêtes SQL / FAISS.
    4. Synthèse finale sans outils.
    5. Préparation des contextes RAGAS.
    """

    print("\n!!! TRAITER_QUESTION_AVEC_TOOLS EXECUTED !!!")
    print("\n--- LLM PLANNING ---")
    print("Question :", question)

    MAX_SQL_QUERIES = 3
    MAX_FAISS_QUERIES = 3
    MAX_GROQ_RETRIES = 2

    # ---------------------------------
    # Appel Groq avec gestion du 429
    # ---------------------------------

    def invoke_groq_with_retry(model, input_messages):

        for attempt in range(MAX_GROQ_RETRIES + 1):

            try:
                return model.invoke(input_messages)

            except Exception as exc:

                error_message = str(exc).lower()

                is_rate_limit = (
                    "429" in error_message
                    or "rate limit" in error_message
                    or "rate_limit_exceeded" in error_message
                )

                if not is_rate_limit or attempt == MAX_GROQ_RETRIES:
                    raise

                match = re.search(
                    r"try again in ([\d.]+)s",
                    str(exc),
                    re.IGNORECASE,
                )

                wait_seconds = (
                    float(match.group(1)) + 1
                    if match
                    else 7
                )

                logging.warning(
                    "Limite Groq atteinte. "
                    "Nouvel essai dans %.1f secondes.",
                    wait_seconds,
                )

                time.sleep(wait_seconds)

    # ---------------------------------
    # Sérialisation
    # ---------------------------------

    def serialize_result(result):

        if isinstance(result, str):
            return result

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    # ---------------------------------
    # Détection des erreurs
    # ---------------------------------

    def is_tool_error(result):

        if isinstance(result, dict):

            return (
                "error" in result
                or "exception" in result
            )

        if isinstance(result, str):

            try:
                parsed = json.loads(result)

                if isinstance(parsed, dict):
                    return (
                        "error" in parsed
                        or "exception" in parsed
                    )

            except json.JSONDecodeError:
                pass

            return result.strip().lower().startswith(
                ("erreur :", "error:", "exception:")
            )

        return False

    # ---------------------------------
    # Vérification du SQL brut
    # ---------------------------------

    def is_raw_sql(value: str):

        return bool(
            re.match(
                r"^\s*(?:SQLQuery\s*:\s*)?"
                r"(SELECT|WITH|INSERT|UPDATE|DELETE|PRAGMA|EXPLAIN)\b",
                value,
                re.IGNORECASE,
            )
        )

    # ---------------------------------
    # Normalisation des objectifs
    # ---------------------------------

    def normalize_text(value):

        value = value.lower().strip()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    # ---------------------------------
    # 1. PLANIFICATION
    # ---------------------------------

    try:

        planner = groq_llm.with_structured_output(
            ResearchPlan,
            method="function_calling",
        )

        plan = invoke_groq_with_retry(
            planner,
            [
                SystemMessage(
                    content=ROUTER_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=question
                ),
            ],
        )

        if not isinstance(plan, ResearchPlan):
            plan = ResearchPlan.model_validate(plan)

        print("\n--- RESEARCH PLAN ---")

        for index, query in enumerate(plan.queries, start=1):

            print(f"\nRecherche {index}")
            print("Outil :", query.tool)
            print("Objectif :", query.objective)
            print("Question :", query.question)

    except Exception as exc:

        logging.exception(
            "Erreur lors de la planification"
        )

        return {
            "answer": f"Erreur de planification : {exc}",
            "contexts": [],
            "source": "error",
        }

    # ---------------------------------
    # 2. VALIDATION DU PLAN
    # ---------------------------------

    sql_count = sum(
        1 for q in plan.queries
        if q.tool == "nba_sql_tool"
    )

    faiss_count = sum(
        1 for q in plan.queries
        if q.tool == "faiss_search_tool"
    )

    if (
        sql_count > MAX_SQL_QUERIES
        or faiss_count > MAX_FAISS_QUERIES
    ):

        return {
            "answer": (
                "Le plan de recherche dépasse le nombre maximal "
                "de requêtes autorisées. Aucune requête n'a été exécutée."
            ),
            "contexts": [],
            "source": "error",
        }

    if not plan.queries:

        return {
            "answer": (
                "Le planificateur n'a identifié aucune recherche "
                "nécessaire. Impossible de produire une réponse fiable."
            ),
            "contexts": [],
            "source": "error",
        }

    # ---------------------------------
    # 3. DÉDUPLICATION DES REQUÊTES
    # ---------------------------------

    unique_queries = []

    seen_objectives = set()
    seen_questions = set()

    for query in plan.queries:

        tool_name = query.tool

        objective_key = (
            tool_name,
            normalize_text(query.objective),
        )

        question_key = (
            tool_name,
            normalize_text(query.question),
        )

        if (
            objective_key in seen_objectives
            or question_key in seen_questions
        ):

            print(
                "\nRecherche redondante supprimée :",
                query.question,
            )

            continue

        if not query.question.strip():

            print("Question vide ignorée.")
            continue

        if tool_name == "nba_sql_tool" and is_raw_sql(
            query.question
        ):

            print(
                "SQL brut refusé dans le plan :",
                query.question,
            )

            continue

        seen_objectives.add(objective_key)
        seen_questions.add(question_key)

        unique_queries.append(query)

    if not unique_queries:

        return {
            "answer": (
                "Aucune requête valide n'a été identifiée."
            ),
            "contexts": [],
            "source": "error",
        }

    print("\n--- VALIDATED PLAN ---")
    print("Nombre de recherches :", len(unique_queries))

    # ---------------------------------
    # 4. EXÉCUTION DES REQUÊTES
    # ---------------------------------

    tool_contexts = []
    sources = []

    for index, query in enumerate(unique_queries, start=1):

        tool_name = query.tool

        print(
            f"\n--- EXECUTION {index}/{len(unique_queries)} ---"
        )

        print("Outil :", tool_name)
        print("Objectif :", query.objective)
        print("Question :", query.question)

        selected_tool = nba_tools_by_name.get(tool_name)

        if selected_tool is None:

            logging.error(
                "Outil inconnu : %s",
                tool_name,
            )

            continue

        try:

            # Les outils reçoivent toujours
            # une question en langage naturel.

            tool_result = selected_tool.invoke(
                {
                    "question": query.question
                }
            )

        except Exception as tool_error:

            logging.exception(
                "Erreur lors de l'exécution de %s",
                tool_name,
            )

            tool_result = {
                "error": str(tool_error)
            }

        print("Tool result :", tool_result)

        if is_tool_error(tool_result):

            print("Résultat en erreur : non conservé.")

            continue

        # Conservation du résultat.
        # Aucun autre appel ne peut le remplacer.

        tool_contexts.append(
            {
                "outil": tool_name,
                "objectif": query.objective,
                "question": query.question,
                "resultat": tool_result,
            }
        )

        if tool_name not in sources:
            sources.append(tool_name)

    # ---------------------------------
    # 5. VÉRIFICATION DES RÉSULTATS
    # ---------------------------------

    if not tool_contexts:

        return {
            "answer": (
                "Aucun outil n'a retourné de résultat exploitable. "
                "Aucune information ne sera inventée."
            ),
            "contexts": [],
            "source": "error",
        }

    print("\n--- ALL TOOL RESULTS ---")

    for context in tool_contexts:
        print(
            "\nOutil :", context["outil"],
            "\nObjectif :", context["objectif"],
            "\nRésultat :", context["resultat"],
        )

    # ---------------------------------
    # 6. SYNTHÈSE FINALE SANS OUTILS
    # ---------------------------------

    FINAL_SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans la NBA.

Tu dois répondre à la question en français.

Tu disposes des résultats des recherches SQL et/ou FAISS.

RÈGLES ABSOLUES :

1. Utilise exclusivement les résultats fournis.

2. Les résultats SQL sont la source de vérité
   pour les statistiques.

3. Reprends exactement les valeurs numériques retournées
   par SQL.

4. N'invente aucune statistique.

5. Ne remplace jamais un résultat SQL par une autre valeur.

6. Ne cherche jamais à vérifier ou corriger un résultat SQL.

7. Ne recalcule pas une statistique déjà fournie.

8. Ne transforme pas une statistique en une autre.

9. Pour les informations documentaires, utilise uniquement
   les éléments présents dans les résultats FAISS.

10. Pour les questions comparatives, confronte les informations
    SQL et FAISS sans inventer de lien de causalité.

11. Si les résultats sont insuffisants, indique-le clairement.

12. Tu ne disposes d'aucun outil et ne dois effectuer
    aucune nouvelle recherche.

Formule une réponse claire, structurée et concise.
"""

    final_messages = [

        SystemMessage(
            content=FINAL_SYSTEM_PROMPT
        ),

        HumanMessage(
            content=(
                f"QUESTION :\n{question}\n\n"
                f"RÉSULTATS DES RECHERCHES :\n"
                f"{json.dumps(tool_contexts, ensure_ascii=False, default=str)}"
            )
        ),
    ]

    try:

        final_message = invoke_groq_with_retry(
            groq_llm,
            final_messages,
        )

    except Exception as exc:

        logging.exception(
            "Erreur lors de la synthèse finale"
        )

        return {
            "answer": f"Erreur de synthèse : {exc}",
            "contexts": [],
            "source": "error",
        }

    # ---------------------------------
    # 7. IDENTIFICATION DE LA SOURCE
    # ---------------------------------

    if (
        "nba_sql_tool" in sources
        and "faiss_search_tool" in sources
    ):

        source = "sql_faiss"

    elif "nba_sql_tool" in sources:

        source = "sql"

    elif "faiss_search_tool" in sources:

        source = "faiss"

    else:

        source = "none"

    # ---------------------------------
    # 8. PRÉPARATION DES CONTEXTES RAGAS
    # ---------------------------------

    contexts = []

    for item in tool_contexts:

        tool_name = item["outil"]
        objective = item["objectif"]
        tool_question = item["question"]
        result = item["resultat"]

        context = (
            f"Objectif : {objective}\n"
            f"Question : {tool_question}\n"
            f"Résultat : {serialize_result(result)}"
        )

        if context not in contexts:
            contexts.append(context)

    # ---------------------------------
    # 9. RETOUR FINAL
    # ---------------------------------

    print("\n--- DONNÉES POUR RAGAS ---")

    print("Question :", question)
    print("Source :", source)
    print("Answer :", final_message.content)
    print("Nombre de contextes :", len(contexts))

    for index, context in enumerate(contexts, start=1):

        print(f"\nContexte {index} :")
        print(context)

    return {
        "answer": str(final_message.content),
        "contexts": contexts,
        "source": source,
    }
    
# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
prompt = st.chat_input(f"Posez votre question sur la {NAME}...")

if prompt:
    query = UserQuery(question=prompt)  # Validation du prompt avec Pydantic
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(query.question)

    # === Début de la logique RAG ===

    # === Traitement de la question avec les outils LangChain ===

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Recherche en cours...")

        with st.spinner("Analyse de la question..."):
            answer = traiter_question_avec_tools(query.question)

        message_placeholder.write(answer)

    # Ajouter la réponse à l'historique Streamlit
    st.session_state.messages.append(
        {"role": "assistant", "content": answer.get("answer", "Aucune réponse fournie.")}
    )


# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")