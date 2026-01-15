"""
Interface Streamlit pour le système RAG Puls-Events
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List

# Configuration de la page
st.set_page_config(
    page_title="Puls-Events - RAG System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_URL = "http://localhost:8000"

from datetime import datetime
import pytz

tz = pytz.timezone("Europe/Paris")
now = datetime.now(tz)

# Format français pour les mois
mois_fr = {
    'January': 'janvier', 'February': 'février', 'March': 'mars', 'April': 'avril',
    'May': 'mai', 'June': 'juin', 'July': 'juillet', 'August': 'août',
    'September': 'septembre', 'October': 'octobre', 'November': 'novembre', 'December': 'décembre'
}
date_str_en = now.strftime("%d %B %Y")
for en, fr in mois_fr.items():
    date_str_en = date_str_en.replace(en, fr)

time_line = f"IMPORTANT - DATE ACTUELLE : Nous sommes le {date_str_en}, il est {now:%H:%M}."
forced_instruction = """
RÈGLE ABSOLUE POUR LES DATES :
- Aujourd'hui = {date_str_en} (format: JJ mois AAAA)
- Un événement en février 2026 est DANS LE FUTUR (après janvier 2026)
- Un événement en janvier 2026 peut être passé ou futur selon le jour
- Toujours vérifier si la date de l'événement est avant ou après {date_str_en}
""".format(date_str_en=date_str_en)

SYSTEM_PROMPT = """
### RÔLE :
Vous êtes l'assistant virtuel officiel de **Puls-Events**, la plateforme web innovante dédiée à la découverte et au suivi en temps réel d'événements culturels.  
Agissez comme un guide culturel numérique accueillant, enthousiaste, réactif et personnalisé.

### OBJECTIF :
Aider les utilisateurs à découvrir, explorer et suivre des événements culturels adaptés à leurs préférences :  
- Rechercher des concerts, spectacles, expositions, festivals, ateliers, conférences, animations patrimoniales, etc.  
- Filtrer par lieu (ville, région, proximité), période (aujourd'hui, ce week-end, ce mois…), type d'événement, gratuit/payant, pour tous publics/enfants…  
- Proposer des suggestions personnalisées en fonction des goûts exprimés  
- Fournir des infos pratiques : dates, horaires, lieux, tarifs, réservation, accessibilité  
- Encourager l'inscription ou le suivi pour recevoir des alertes/notifications

### SOURCES AUTORISÉES :
- Données agrégées via la plateforme Puls-Events (collectées depuis OpenAgenda et sources partenaires officielles)
- Utilise UNIQUEMENT les informations présentes dans le contexte fourni
- Si l'information n'est pas dans le contexte, indique-le clairement

### COMPORTEMENT & STYLE :
Ton : Chaleureux, enthousiaste, convivial, moderne et accessible  
Précision : Informations exactes, à jour et uniquement tirées des données de la plateforme  
Personnalisation : Poser des questions pour affiner les suggestions
Enthousiasme : Valoriser la richesse culturelle sans exagération
Ambiguïté : Demander poliment des précisions si nécessaire
Info Manquante : Si l'événement n'est pas dans le contexte, l'indiquer clairement

### RÈGLES IMPORTANTES :
- Rester factuel : utiliser uniquement les informations du contexte fourni
- Être descriptif plutôt que subjectif (éviter "incontournable", "génial")
- DATES : Comparer attentivement les dates des événements avec la date actuelle fournie en début de contexte

### EXEMPLE D'INTERACTION GUIDÉE :
Utilisateur : « Quels événements sympas ce week-end à Paris ? »
Assistant Attendu :
« Voici ce que Puls-Events a en stock pour ce week-end à Paris :
• Vendredi 14 janv. 20h – Théâtre de l'Athénée : Concert jazz « Vibes d'hiver » – Tarif 18 € / 12 € réduit
• Samedi 15 janv. 14h-18h – Musée Guimet : Atelier famille « Manga, tout un art! » – Gratuit sur inscription
• Dimanche 16 janv. 11h – Comédie-Française : Représentation théâtrale – Entrée à partir de 25€
Tu préfères quelque chose de gratuit, en intérieur, ou plutôt musical ? Je peux affiner selon tes goûts ! 😊
Retrouve tous les détails et réserve sur la plateforme Puls-Events. »


""" + "\n\n" + time_line + "\n" + forced_instruction


# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-top: 0;
    }
    .source-card {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .metadata-badge {
        display: inline-block;
        background-color: #e8eaf6;
        padding: 0.2rem 0.6rem;
        margin: 0.2rem;
        border-radius: 12px;
        font-size: 0.85rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None


def check_api_health() -> Dict:
    """Vérifie l'état de l'API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "API timeout (>10s) - Le service peut être occupé"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Impossible de se connecter à l'API - Vérifiez que le conteneur est démarré"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


def query_rag_system(
    question: str,
    top_k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9
) -> Dict:
    """Envoie une requête au système RAG"""
    try:
        # IMPORTANT: Send question and system_prompt SEPARATELY
        # This allows the RAG pipeline to:
        # 1. Use only the question for vector search (semantic matching)
        # 2. Use the system_prompt only for LLM generation (response formatting)
        payload = {
            "question": question,  # Raw question for vector search
            "top_k": top_k,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "system_prompt": SYSTEM_PROMPT  # Separate system prompt for LLM
        }

        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=120  # Augmenté à 120s pour gérer les retries de l'API Mistral
        )

        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        elif response.status_code == 503:
            return {
                "status": "error",
                "error": "Service temporairement indisponible. L'API Mistral est surchargée, veuillez réessayer dans quelques instants."
            }
        else:
            return {
                "status": "error",
                "error": f"API returned status code {response.status_code}: {response.text}"
            }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Timeout de la requête (>120s). Cela peut indiquer que l'API Mistral est temporairement indisponible. Veuillez réessayer."
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error": "Impossible de se connecter à l'API. Vérifiez que le service est en cours d'exécution."
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"Erreur de requête: {str(e)}"}


def display_source(source: Dict, index: int):
    """Affiche une source de manière formatée"""
    with st.expander(f"📄 Source {index + 1}: {source.get('title', 'Document')} - Score: {source.get('score', 0):.2f}"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Fichier:** `{source.get('file', 'N/A')}`")
            st.markdown(f"**Extrait:**")
            st.markdown(f"_{source.get('excerpt', 'N/A')}_")

        with col2:
            st.metric("Score de similarité", f"{source.get('score', 0):.3f}")


def display_metadata(metadata: Dict):
    """Affiche les métadonnées de la requête"""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("⏱️ Temps de requête", f"{metadata.get('query_time_ms', 0):.0f} ms")

    with col2:
        st.metric("📚 Documents consultés", metadata.get('documents_searched', 0))


# En-tête de l'application
st.markdown('<p class="main-header">🎭 Puls-Events</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Événements culturels en temps réel</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")

    # Paramètres de recherche
    st.subheader("🔍 Paramètres de recherche")

    top_k = st.slider(
        "Nombre de documents (top_k)",
        min_value=1,
        max_value=20,
        value=5,
        help="Nombre de documents similaires à récupérer"
    )

    st.markdown("---")

    # Paramètres du modèle LLM
    st.subheader("🤖 Paramètres du modèle")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Contrôle la créativité des réponses. 0 = déterministe, 1 = très créatif"
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=128,
        max_value=2048,
        value=500,
        step=64,
        help="Nombre maximum de tokens dans la réponse"
    )

    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.93,
        step=0.01,
        help="Contrôle la diversité des réponses via nucleus sampling"
    )
    st.markdown("---")

    # Bouton pour réinitialiser l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")

    # Affichage du system prompt
    st.subheader("💬 System Prompt")
    st.text_area(
        "Prompt système (lecture seule)",
        value=SYSTEM_PROMPT,
        height=100,
        disabled=True,
        help="Le prompt système est utilisé pour contextualiser toutes les questions"
    )

    st.markdown("---")

    # Paramètres de L’API
    st.subheader("Paramètres de l’API")

    # Vérification de l'état de l'API
    if st.button("🔄 Vérifier l'API"):
        with st.spinner("Vérification..."):
            st.session_state.api_status = check_api_health()

    if st.session_state.api_status:
        if st.session_state.api_status["status"] == "healthy":
            st.success("✅ API opérationnelle")
            data = st.session_state.api_status.get("data", {})
            if "index_size" in data:
                st.info(f"📊 Taille de l'index: {data['index_size']} documents")
        else:
            st.error(f"❌ API non disponible: {st.session_state.api_status.get('error', 'Unknown error')}")

    # Avertissement pour les problèmes Mistral API
    with st.expander("⚠️ En cas de problème", expanded=False):
        st.warning("""
        **Si vous rencontrez des timeouts ou des erreurs:**

        - L'API Mistral peut être temporairement surchargée (erreur 503)
        - Attendez quelques secondes et réessayez
        - Les requêtes peuvent prendre jusqu'à 2 minutes en cas de charge élevée
        - Le système fait automatiquement plusieurs tentatives
        """)

    st.markdown("---")

    # Informations
    st.subheader("ℹ️ Informations")
    st.caption(f"**Messages dans l'historique:** {len(st.session_state.messages)}")
    st.caption(f"**API Endpoint:** {API_URL}")

# Zone principale - Historique de conversation
st.subheader("💬 Conversation")

# Affichage de l'historique
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown(f'<div class="user-message"><strong>👤 Vous:</strong><br>{message["content"]}</div>',
                       unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{message["content"]}</div>',
                       unsafe_allow_html=True)

            # Affichage des sources si disponibles
            if "sources" in message and message["sources"]:
                st.markdown("**📚 Sources utilisées:**")
                for idx, source in enumerate(message["sources"]):
                    display_source(source, idx)

            # Affichage des métadonnées si disponibles
            if "metadata" in message and message["metadata"]:
                with st.expander("📊 Métadonnées de la requête"):
                    display_metadata(message["metadata"])

# Zone de saisie de la question
st.markdown("---")
question = st.chat_input("Posez votre question sur les événements culturels...")

if question:
    # Ajout de la question à l'historique
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat()
    })

    # Affichage immédiat de la question
    with st.container():
        st.markdown(f'<div class="user-message"><strong>👤 Vous:</strong><br>{question}</div>',
                   unsafe_allow_html=True)

    # Requête au système RAG
    with st.spinner("🔍 Recherche en cours..."):
        result = query_rag_system(
            question,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

    if result["status"] == "success":
        data = result["data"]
        answer = data.get("answer", "Aucune réponse générée")
        sources = data.get("sources", [])
        metadata = data.get("metadata", {})

        # Ajout de la réponse à l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })

        # Affichage de la réponse
        with st.container():
            st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{answer}</div>',
                       unsafe_allow_html=True)

            # Affichage des sources
            if sources:
                st.markdown("**📚 Sources utilisées:**")
                for idx, source in enumerate(sources):
                    display_source(source, idx)

            # Affichage des métadonnées
            if metadata:
                with st.expander("📊 Métadonnées de la requête"):
                    display_metadata(metadata)

        st.success("✅ Réponse générée avec succès!")
    else:
        error_msg = result.get("error", "Erreur inconnue")
        st.error(f"❌ Erreur lors de la requête: {error_msg}")

        # Ajout de l'erreur à l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Erreur: {error_msg}",
            "timestamp": datetime.now().isoformat()
        })

# Footer
st.markdown("---")
st.caption("🎭 Puls-Events - Système RAG avec Docling, FAISS et LangChain | Propulsé par Mistral AI")
