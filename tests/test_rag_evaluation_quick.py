"""
Test rapide de l'évaluation Ragas (2 questions seulement).

Ce script est une version simplifiée pour tester rapidement que Ragas fonctionne
avec le LLM configuré.

Exécution :
    python tests/test_rag_evaluation_quick.py
"""

import sys
from pathlib import Path

# Import du module principal
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.test_rag_evaluation import (
    check_api_health,
    query_rag_system,
    extract_contexts_from_response,
    evaluate_with_ragas,
    display_results,
    TEST_DATASET
)


def main():
    """Test rapide avec 2 questions."""
    print("=" * 70)
    print("TEST RAPIDE DE L'ÉVALUATION RAGAS (2 questions)")
    print("=" * 70)

    # 1. Vérifier l'API
    if not check_api_health():
        print("\n✗ API non accessible")
        return 1

    # 2. Sélectionner 2 questions de test (1 simple + 1 complexe)
    quick_tests = [
        TEST_DATASET[0],  # Question simple
        TEST_DATASET[5],  # Question complexe
    ]

    print(f"\nQuestions sélectionnées:")
    for i, test in enumerate(quick_tests, 1):
        print(f"  {i}. [{test['category']}] {test['question'][:60]}...")

    # 3. Collecter les réponses
    print("\n" + "=" * 70)
    print("COLLECTE DES RÉPONSES")
    print("=" * 70)

    dataset = []
    for i, test_case in enumerate(quick_tests, 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        category = test_case['category']

        print(f"\n[{i}/{len(quick_tests)}] {question}")

        response = query_rag_system(question, top_k=3)

        if response:
            answer = response.get('answer', '')
            contexts = extract_contexts_from_response(response)

            print(f"✓ Réponse: {len(answer)} caractères")
            print(f"✓ Contextes: {len(contexts)}")

            dataset.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "category": category
            })
        else:
            print(f"✗ Échec")

    if not dataset:
        print("\n✗ Aucune réponse collectée")
        return 1

    # 4. Évaluer avec Ragas
    results = evaluate_with_ragas(dataset)

    if not results:
        print("\n✗ Évaluation Ragas échouée")
        return 1

    # 5. Afficher les résultats
    display_results(results, dataset)

    print("\n" + "=" * 70)
    print("TEST RAPIDE TERMINÉ")
    print("=" * 70)
    print("\nPour une évaluation complète (10 questions):")
    print("  python tests/test_rag_evaluation.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
