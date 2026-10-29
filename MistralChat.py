import streamlit as st
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from dotenv import load_dotenv
from sql.sql_tool import create_sql_tool

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME, DASHSCOPE_API_KEY
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

# Configuration de l'observabilité en premier
import logfire

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

# --- Configuration de l'API Mistral / Qwen3.8 ---
api_key = DASHSCOPE_API_KEY
model = MODEL_NAME

# --- Initialisation du modèle Qwen3.8-flash via Pydantic AI ---
qwen_model = OpenAIChatModel(
    model_name=model,
    provider=OpenAIProvider(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    ),
)

if not api_key:
    st.error("Erreur : Clé API Qwen non trouvée (DASHSCOPE_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

# Prompt pour l'agent Pydantic AI
rag_agent = Agent(
    qwen_model,
    output_type=RAGResponse,
    model_settings=ModelSettings(extra_body={"enable_thinking": False}),
    system_prompt=(
        "Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA. "
        "Tu réponds aux questions des fans en te basant uniquement sur le contexte "
        "fourni par la base de connaissances. "
        "Si le contexte ne permet pas de répondre, indique-le clairement."
    ),
)

# Modèle LangChain : sélection et appel des outils
qwen_langchain = ChatOpenAI(
    model=model,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0,
    extra_body={"enable_thinking": False},
)

# --- Configuration SQL ---
ask_database = create_sql_tool(qwen_langchain)

# try:
#     #client = MistralClient(api_key=api_key)
#     client = ChatOpenAI(
#         model=model,
#         api_key=api_key,
#         base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
#         temperature=0.2,
#         extra_body={"enable_thinking": False}
#     )
#     logging.info("Client Qwen3.8-flash initialisé.")
# except Exception as e:
#     st.error(f"Erreur lors de l'initialisation du client Qwen : {e}")
#     logging.exception("Erreur initialisation client Qwen")
#     st.stop()

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

# --- Prompt Système pour RAG ---
# Adaptez ce prompt selon vos besoins
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


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

# Le modèle sait désormais appeler les outils déclarés.
qwen_with_tools = qwen_langchain.bind_tools(
    nba_tools,
    parallel_tool_calls=False,
)


# --- Routage et synthèse avec LangChain ---

ROUTER_SYSTEM_PROMPT = """
Tu es NBA Analyst AI, un assistant expert sur la NBA.

Pour répondre à chaque question, choisis exactement un outil :

- nba_sql_tool :
  pour les statistiques chiffrées, classements, moyennes,
  totaux, comparaisons de joueurs et statistiques d'équipes.

- faiss_search_tool :
  pour les questions explicatives, qualitatives, historiques
  ou nécessitant des informations issues de la base documentaire.

Appelle l'outil approprié avant de répondre.
Ne réponds pas à partir de tes seules connaissances si un outil
peut fournir les informations nécessaires.
N'invente aucune statistique ni aucun résultat.

Après réception du résultat de l'outil, rédige une réponse
claire en français, en te fondant sur ce résultat.
Si l'outil renvoie une erreur ou ne trouve pas l'information,
explique-le explicitement.
"""


def traiter_question_avec_tools(question: str) -> str:
    """Choisit un outil, l'exécute, puis synthétise sa réponse."""

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    try:
        # Appel 1 : Qwen choisit un outil.
        ai_message = qwen_with_tools.invoke(messages)
        messages.append(ai_message)

        # Si aucun outil n'est appelé, retourner la réponse du modèle.
        if not ai_message.tool_calls:
            return ai_message.content or (
                "Je n'ai pas pu déterminer comment traiter la question."
            )

        # Exécuter les outils demandés.
        # Le prompt demande un seul outil ; cette boucle gère aussi
        # correctement le cas où le modèle en demande plusieurs.
        for tool_call in ai_message.tool_calls:
            selected_tool = nba_tools_by_name.get(tool_call["name"])

            if selected_tool is None:
                tool_result = (
                    f"Erreur : outil inconnu {tool_call['name']}."
                )
            else:
                tool_result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

        # Appel final : Qwen synthétise les résultats des outils.
        final_message = qwen_langchain.invoke(messages)

        if isinstance(final_message.content, str):
            return final_message.content

        return str(final_message.content)

    except Exception as exc:
        logging.exception("Erreur dans le routage LangChain")
        return f"Une erreur technique est survenue : {exc}"


# --- Fonctions ---

def generer_reponse(question: str, context_str: str) -> RAGResponse:
    """
    Génère une réponse structurée avec Pydantic AI.
    """
    if not question:
        logging.warning("Tentative de génération avec une question vide.")
        return RAGResponse(
            answer="Je ne peux pas traiter une demande vide."
        )

    try:
        logging.info(
            f"Appel à Qwen via Pydantic AI, modèle '{model}'."
        )

        prompt = (
            "CONTEXTE DE LA BASE DE CONNAISSANCES\n\n"
            f"{context_str}\n\n"
            "QUESTION DU FAN\n\n"
            f"{question}"
        )

        with logfire.span("RAG  - génération de réponse LLM"):
            result = rag_agent.run_sync(prompt)
            logfire.info("Réponse générée par l'agent Pydantic AI",  output_type=type(result.output).__name__,)

            return result.output
    
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Qwen: {e}")
        logging.exception("Erreur API Qwen pendant client.chat")
        return RAGResponse(
            answer="Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard.")

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
        {"role": "assistant", "content": answer}
    )


# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")