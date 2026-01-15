"""
Tests d'évaluation de la qualité du système RAG avec Ragas.

Ce fichier utilise le framework Ragas pour évaluer la pertinence et la qualité
des réponses générées par le système RAG sur un jeu de données de test.

Métriques évaluées :
- Faithfulness : La réponse est-elle fidèle au contexte récupéré ?
- Answer Relevancy : La réponse est-elle pertinente par rapport à la question ?
- Context Precision : Les contextes récupérés sont-ils pertinents ?
- Context Recall : Le contexte contient-il toutes les informations nécessaires ?

Exécution :
    python tests/test_rag_evaluation.py
    OU
    docker compose -f docker/docker-compose.yml exec rag-system python tests/test_rag_evaluation.py

Prérequis :
    - L'API doit être en cours d'exécution (http://localhost:8000)
    - Les documents doivent être indexés
    - Variables d'environnement : OPENAI_API_KEY ou autre LLM configuré
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Configuration
API_BASE_URL = os.environ.get('RAG_API_URL', 'http://localhost:8000')
API_TIMEOUT = 60  # secondes (augmenté pour les requêtes complexes)


# ============================================================================
# JEU DE DONNÉES DE TEST (10 cas test)
# ============================================================================

TEST_DATASET = [
    {
        "question": "Quels sont les événements musicaux prévus en janvier 2026 à Paris ?",
        "ground_truth": "Il y a plusieurs événements musicaux en janvier 2026, notamment 'LE BALCON : LES LUNDIS MUSICAUX' avec Julie Fuchs le 26 janvier 2026.",
        "category": "simple",
        "description": "Question simple sur un type d'événement et une période"
    },
    {
        "question": "Y a-t-il des spectacles de théâtre ou des représentations théâtrales disponibles ?",
        "ground_truth": "Oui, il y a des représentations théâtrales comme 'Lettres non-écrites' de la Compagnie Lieux-dits et 'Moi, Elles' de la Cie Abricotier d'Argent.",
        "category": "simple",
        "description": "Question sur un type d'événement spécifique"
    },
    {
        "question": "Où se déroule l'événement 'Lettres non-écrites' ?",
        "ground_truth": "L'événement 'Lettres non-écrites' se déroule à Paris, dans le 10ème arrondissement (code postal 75010).",
        "category": "simple",
        "description": "Question sur le lieu d'un événement spécifique"
    },
    {
        "question": "Quand commence l'événement 'LE BALCON : LES LUNDIS MUSICAUX - Julie Fuchs & invités' ?",
        "ground_truth": "L'événement commence le 26 janvier 2026 à 19h00.",
        "category": "simple",
        "description": "Question sur la date/heure d'un événement"
    },
    {
        "question": "Quels événements sont prévus en mai 2026 ?",
        "ground_truth": "En mai 2026, il y a notamment l'événement 'Lettres non-écrites' prévu le 12 mai 2026.",
        "category": "simple",
        "description": "Question sur une période spécifique"
    },
    {
        "question": "Y a-t-il des événements liés à la cybersécurité ou à la technologie ?",
        "ground_truth": "Oui, il y a un événement 'SAVE THE DATE - Cybersécurité - Quelles innovations en Europe' organisé par le Cercle Cyber de l'Institut G9+, prévu le 11 février 2026.",
        "category": "complexe",
        "description": "Question thématique nécessitant une recherche sémantique"
    },
    {
        "question": "Quels sont les événements culturels gratuits disponibles ?",
        "ground_truth": "Les informations sur la gratuité des événements ne sont pas systématiquement disponibles dans les données. Il faudrait vérifier les détails de chaque événement individuellement.",
        "category": "complexe",
        "description": "Question nécessitant une information qui pourrait être absente"
    },
    {
        "question": "Peux-tu me donner des informations sur les événements avec Erik Satie au programme ?",
        "ground_truth": "Oui, l'événement 'LE BALCON : LES LUNDIS MUSICAUX - Julie Fuchs & invités' le 26 janvier 2026 est consacré à la figure d'Erik Satie.",
        "category": "complexe",
        "description": "Question nécessitant une recherche dans les descriptions"
    },
    {
        "question": "Existe-t-il des événements organisés par la Compagnie Lieux-dits ?",
        "ground_truth": "Oui, la Compagnie Lieux-dits organise l'événement 'Lettres non-écrites' de David Geselson.",
        "category": "simple",
        "description": "Question sur un organisateur spécifique"
    },
    {
        "question": "Quel temps fera-t-il demain à Paris ?",
        "ground_truth": "Cette question concerne la météo et n'est pas liée aux événements culturels. Le système ne devrait pas pouvoir y répondre avec les données disponibles.",
        "category": "piège",
        "description": "Question hors-sujet pour tester les limites du système"
    }
]


# ============================================================================
# FONCTIONS D'INTERACTION AVEC L'API
# ============================================================================

def check_api_health() -> bool:
    """
    Vérifie que l'API est accessible et fonctionnelle.

    Returns:
        True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API accessible - Index size: {data.get('index_size', 'N/A')} vecteurs")
            return True
        else:
            print(f"✗ API répond avec le code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Impossible de contacter l'API: {e}")
        return False


def query_rag_system(question: str, top_k: int = 5) -> Optional[Dict]:
    """
    Interroge le système RAG via l'API.

    Args:
        question: La question à poser
        top_k: Nombre de contextes à récupérer

    Returns:
        Dictionnaire contenant la réponse, les sources et les métadonnées
        None en cas d'erreur
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur API pour la question '{question}': {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion pour la question '{question}': {e}")
        return None


def extract_contexts_from_response(response: Dict) -> List[str]:
    """
    Extrait les contextes (chunks) de la réponse de l'API.

    Args:
        response: Réponse de l'API

    Returns:
        Liste des textes des contextes récupérés
    """
    contexts = []
    sources = response.get('sources', [])

    for source in sources:
        excerpt = source.get('excerpt', '')
        if excerpt:
            # Nettoyer les "..." ajoutés à la fin
            excerpt = excerpt.rstrip('.')
            contexts.append(excerpt)

    return contexts


# ============================================================================
# COLLECTE DES DONNÉES POUR RAGAS
# ============================================================================

def collect_rag_responses() -> List[Dict]:
    """
    Collecte les réponses du système RAG pour toutes les questions de test.

    Returns:
        Liste de dictionnaires avec question, answer, contexts, ground_truth
    """
    print("\n" + "=" * 70)
    print("COLLECTE DES RÉPONSES DU SYSTÈME RAG")
    print("=" * 70)

    dataset = []

    for i, test_case in enumerate(TEST_DATASET, 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        category = test_case['category']

        print(f"\n[{i}/{len(TEST_DATASET)}] Catégorie: {category}")
        print(f"Question: {question}")

        # Interroger l'API
        response = query_rag_system(question, top_k=5)

        if response:
            answer = response.get('answer', '')
            contexts = extract_contexts_from_response(response)

            print(f"✓ Réponse obtenue ({len(answer)} caractères)")
            print(f"✓ Contextes récupérés: {len(contexts)}")

            dataset.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "category": category
            })
        else:
            print(f"✗ Échec de la récupération de la réponse")

    print("\n" + "=" * 70)
    print(f"Collecte terminée: {len(dataset)}/{len(TEST_DATASET)} réponses obtenues")
    print("=" * 70)

    return dataset


# ============================================================================
# ÉVALUATION AVEC RAGAS
# ============================================================================

def evaluate_with_ragas(dataset: List[Dict]) -> Dict:
    """
    Évalue le système RAG avec Ragas.

    Args:
        dataset: Liste de dictionnaires avec question, answer, contexts, ground_truth

    Returns:
        Dictionnaire contenant les scores et métriques
    """
    try:
        # Import Ragas (install avec: pip install ragas)
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

    except ImportError as e:
        print(f"\n✗ ERREUR: Ragas n'est pas installé.")
        print(f"  Installez-le avec: pip install ragas")
        print(f"  Erreur: {e}")
        return None

    print("\n" + "=" * 70)
    print("ÉVALUATION AVEC RAGAS")
    print("=" * 70)

    # Charger les paramètres depuis la configuration
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.config import get_settings
        settings = get_settings()
        model_name = settings.model_name
        print(f"\nModèle LLM configuré: {model_name}")
    except Exception as e:
        print(f"⚠ Impossible de charger la configuration: {e}")
        model_name = os.environ.get('MODEL_NAME', 'gpt-4-turbo-preview')
        print(f"Utilisation du modèle par défaut: {model_name}")

    # Configurer le LLM et les embeddings pour Ragas selon le modèle configuré
    llm = None
    embeddings = None

    try:
        model_lower = model_name.lower()

        # Anthropic Claude
        if "claude" in model_lower or "anthropic" in model_lower:
            from langchain_anthropic import ChatAnthropic
            from langchain_huggingface import HuggingFaceEmbeddings
            print(f"Configuration de Ragas avec Anthropic Claude")
            llm = ChatAnthropic(model=model_name, temperature=0)
            # Utiliser HuggingFace embeddings (gratuit)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Mistral AI
        elif "mistral" in model_lower and "-" in model_lower:
            from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
            print(f"Configuration de Ragas avec Mistral AI")
            llm = ChatMistralAI(model=model_name, temperature=0)
            # Utiliser Mistral embeddings (cohérent avec le LLM)
            try:
                embeddings = MistralAIEmbeddings(model="mistral-embed")
                print(f"  Embeddings: mistral-embed")
            except Exception as e:
                print(f"  ⚠ Erreur Mistral embeddings: {e}, utilisation HuggingFace")
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # OpenAI
        else:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            print(f"Configuration de Ragas avec OpenAI")
            llm = ChatOpenAI(model=model_name, temperature=0)
            embeddings = OpenAIEmbeddings()

    except ImportError as e:
        print(f"⚠ Impossible d'importer le LLM provider: {e}")
        print(f"  Ragas utilisera la configuration par défaut")

    # Convertir en Dataset Hugging Face
    hf_dataset = Dataset.from_list(dataset)

    print(f"\nDataset créé: {len(hf_dataset)} exemples")
    print(f"Colonnes: {hf_dataset.column_names}")

    # Définir les métriques à calculer
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print(f"\nMétriques à calculer:")
    print(f"  - Faithfulness (fidélité)")
    print(f"  - Answer Relevancy (pertinence de la réponse)")
    print(f"  - Context Precision (précision du contexte)")
    print(f"  - Context Recall (rappel du contexte)")

    print(f"\n⏳ Évaluation en cours (cela peut prendre quelques minutes)...")

    try:
        # Exécuter l'évaluation avec le LLM et embeddings configurés
        eval_kwargs = {
            "dataset": hf_dataset,
            "metrics": metrics,
        }

        # Ajouter le LLM si configuré
        if llm:
            eval_kwargs["llm"] = llm

        # Ajouter les embeddings si configurés
        if embeddings:
            eval_kwargs["embeddings"] = embeddings

        results = evaluate(**eval_kwargs)

        return results

    except Exception as e:
        print(f"\n✗ ERREUR pendant l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================================

def display_results(results, dataset: List[Dict]):
    """
    Affiche les résultats de l'évaluation de manière lisible.

    Args:
        results: Résultats de Ragas (EvaluationResult object)
        dataset: Dataset utilisé pour l'évaluation
    """
    print("\n" + "=" * 70)
    print("RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("=" * 70)

    if not results:
        print("\n✗ Aucun résultat disponible")
        return

    # Convertir l'objet EvaluationResult en dictionnaire
    try:
        # Ragas 0.2+ retourne un objet avec les scores directement accessibles
        if hasattr(results, 'to_pandas'):
            results_df = results.to_pandas()
            # Les scores des métriques sont dans des colonnes spécifiques
            # On calcule la moyenne pour chaque métrique numérique
            results_dict = {}
            for col in results_df.columns:
                if col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    # Extraire les valeurs numériques et calculer la moyenne
                    try:
                        values = results_df[col].dropna()
                        if len(values) > 0:
                            results_dict[col] = float(values.mean())
                    except:
                        pass
        elif hasattr(results, '__dict__'):
            results_dict = {k: v for k, v in results.__dict__.items()
                          if k in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']}
        else:
            results_dict = dict(results)

        # Si aucune métrique n'a été extraite, essayer une autre méthode
        if not results_dict:
            # Essayer d'accéder aux attributs directement
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                if hasattr(results, metric):
                    value = getattr(results, metric)
                    if isinstance(value, (int, float)):
                        results_dict[metric] = value

    except Exception as e:
        print(f"⚠ Impossible de convertir les résultats: {e}")
        results_dict = {}

    # Debug : afficher la structure si aucune métrique n'est trouvée
    if not results_dict:
        print(f"\n⚠ Debug - Type de results: {type(results)}")
        print(f"⚠ Debug - Attributs disponibles: {dir(results)}")
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            print(f"⚠ Debug - Colonnes du DataFrame: {df.columns.tolist()}")
            print(f"⚠ Debug - Premières lignes:\n{df.head()}")

    # Afficher les scores globaux
    print("\n### Scores globaux (0 à 1, plus élevé = meilleur) ###\n")

    metrics_info = {
        'faithfulness': 'Fidélité au contexte',
        'answer_relevancy': 'Pertinence de la réponse',
        'context_precision': 'Précision du contexte',
        'context_recall': 'Rappel du contexte'
    }

    for metric_name, description in metrics_info.items():
        score = results_dict.get(metric_name, 'N/A')
        if isinstance(score, (float, int)):
            # Barre de progression visuelle
            bar_length = 30
            filled = int(score * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"{description:30} [{bar}] {score:.3f}")
        else:
            print(f"{description:30} : {score}")

    # Score moyen
    numeric_scores = [v for v in results_dict.values() if isinstance(v, (int, float))]
    if numeric_scores:
        avg_score = sum(numeric_scores) / len(numeric_scores)
        print(f"\n{'Score moyen':30} : {avg_score:.3f}")

    # Analyse par catégorie
    print("\n### Analyse par catégorie ###\n")

    categories = {}
    for item in dataset:
        cat = item.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    for category, items in categories.items():
        print(f"{category.capitalize():15} : {len(items)} question(s)")

    # Recommandations
    print("\n### Recommandations ###\n")

    if numeric_scores:
        if results_dict.get('faithfulness', 1.0) < 0.7:
            print("⚠ Faithfulness faible : Les réponses ne sont pas toujours fidèles au contexte.")
            print("  → Améliorer le prompt système pour insister sur l'utilisation du contexte")

        if results_dict.get('answer_relevancy', 1.0) < 0.7:
            print("⚠ Answer Relevancy faible : Les réponses manquent de pertinence.")
            print("  → Améliorer le prompt système et la qualité des chunks")

        if results_dict.get('context_precision', 1.0) < 0.7:
            print("⚠ Context Precision faible : Les contextes récupérés ne sont pas pertinents.")
            print("  → Améliorer l'indexation, essayer un autre modèle d'embedding")

        if results_dict.get('context_recall', 1.0) < 0.7:
            print("⚠ Context Recall faible : Le contexte ne contient pas toutes les infos nécessaires.")
            print("  → Augmenter le nombre de chunks récupérés (top_k)")

        if avg_score >= 0.8:
            print("✓ Excellente performance globale du système RAG !")
        elif avg_score >= 0.6:
            print("✓ Performance acceptable, des améliorations sont possibles")
        else:
            print("⚠ Performance faible, des améliorations significatives sont nécessaires")


# ============================================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================================

def save_results(dataset: List[Dict], results, output_dir: Path):
    """
    Sauvegarde les résultats dans des fichiers JSON.

    Args:
        dataset: Dataset utilisé
        results: Résultats de l'évaluation (EvaluationResult object)
        output_dir: Répertoire de sortie
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sauvegarder le dataset
    dataset_file = output_dir / f"rag_evaluation_dataset_{timestamp}.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Convertir les résultats en dictionnaire pour la sauvegarde
    try:
        if hasattr(results, 'to_pandas'):
            results_df = results.to_pandas()
            # Extraire uniquement les colonnes de métriques
            results_dict = {}
            for col in results_df.columns:
                if col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    try:
                        values = results_df[col].dropna()
                        if len(values) > 0:
                            results_dict[col] = float(values.mean())
                    except:
                        pass
        elif hasattr(results, '__dict__'):
            results_dict = {k: v for k, v in results.__dict__.items() if not k.startswith('_')}
        else:
            results_dict = dict(results) if results else {}
    except Exception as e:
        print(f"⚠ Erreur lors de la conversion des résultats: {e}")
        results_dict = {}

    # Sauvegarder les résultats
    results_file = output_dir / f"rag_evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Résultats sauvegardés:")
    print(f"  - Dataset: {dataset_file}")
    print(f"  - Résultats: {results_file}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale pour exécuter l'évaluation complète.
    """
    print("=" * 70)
    print("ÉVALUATION DE LA QUALITÉ DU SYSTÈME RAG AVEC RAGAS")
    print("=" * 70)
    print(f"API URL: {API_BASE_URL}")
    print(f"Nombre de questions de test: {len(TEST_DATASET)}")
    print("=" * 70)

    # 1. Vérifier que l'API est accessible
    if not check_api_health():
        print("\n✗ ÉCHEC : L'API n'est pas accessible.")
        print("  Assurez-vous que le serveur est démarré:")
        print("  docker compose -f docker/docker-compose.yml up -d")
        return 1

    # 2. Collecter les réponses du système RAG
    dataset = collect_rag_responses()

    if not dataset:
        print("\n✗ ÉCHEC : Aucune réponse n'a pu être collectée.")
        return 1

    # 3. Évaluer avec Ragas
    results = evaluate_with_ragas(dataset)

    if not results:
        print("\n✗ ÉCHEC : L'évaluation Ragas a échoué.")
        print("  Vérifiez que Ragas est installé et qu'une clé API LLM est configurée.")
        return 1

    # 4. Afficher les résultats
    display_results(results, dataset)

    # 5. Sauvegarder les résultats
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "tests" / "results"
    save_results(dataset, results, output_dir)

    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
