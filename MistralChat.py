# MistralChat.py (version RAG)
import streamlit as st
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from dotenv import load_dotenv

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

# --- Configuration de l'API Mistral ---
api_key = DASHSCOPE_API_KEY
model = MODEL_NAME

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

    # 2. Vérifier si le Vector Store est disponible
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        st.stop()

    # 3. Rechercher le contexte dans le Vector Store
    try:
        with logfire.span(
            "RAG - recherche FAISS",
            search_k=SEARCH_K,
        ):
            search_results = [
                SearchResult(**result)
                for result in vector_store_manager.search(
                    query.question,
                    k=SEARCH_K,
                )
            ]

            logfire.info(
                "Recherche terminée",
                result_count=len(search_results),
                scores=[result.score for result in search_results],
            )

    except Exception as e:
        logging.exception("Erreur pendant la recherche FAISS")
        search_results = []

    # try:
    #     logging.info(f"Recherche de contexte pour la question: '{query.question}' avec k={SEARCH_K}")
        
    #     search_results = [
    #         SearchResult(**results)
    #         for results in vector_store_manager.search(query.question, k=SEARCH_K)
    #     ] 
    #     logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
    # except Exception as e:
    #     st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
    #     logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {query.question}")
    #     search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
    context_str = "\n\n---\n\n".join(
        f"Source: {res.metadata.get('source', 'Inconnue')}"
        f"Score: {res.score:.1f}%)\n"
        f"Contenu: {res.text}"
    for res in search_results)

    if not search_results:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {query.question}")

    # === Fin de la logique RAG ===


    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt formaté
        rag_reponse = generer_reponse(
            question=query.question,
            context_str=context_str
        )

        # Affichage de la réponse complète
        message_placeholder.write(rag_reponse.answer)

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": rag_reponse.answer})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")