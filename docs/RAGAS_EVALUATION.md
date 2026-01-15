# Évaluation RAG avec Ragas

Ce document explique comment utiliser Ragas pour évaluer la qualité et la pertinence du système RAG.

## Table des matières

1. [Introduction à Ragas](#introduction-à-ragas)
2. [Métriques évaluées](#métriques-évaluées)
3. [Jeu de données de test](#jeu-de-données-de-test)
4. [Installation](#installation)
5. [Exécution des tests](#exécution-des-tests)
6. [Interprétation des résultats](#interprétation-des-résultats)
7. [Amélioration du système](#amélioration-du-système)

---

## Introduction à Ragas

[Ragas](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) est un framework open-source pour évaluer les systèmes RAG. Il fournit des métriques standardisées pour mesurer la qualité des réponses générées.

### Pourquoi évaluer avec Ragas ?

- **Objectivité** : Métriques quantitatives basées sur des LLMs
- **Complétude** : Évalue plusieurs aspects du système RAG
- **Standardisation** : Permet de comparer différentes configurations
- **Détection des problèmes** : Identifie les points faibles du système

---

## Métriques évaluées

### 1. Faithfulness (Fidélité)

**Définition** : Mesure si la réponse générée est fidèle au contexte récupéré.

**Calcul** :
```
Faithfulness = nombre_affirmations_supportées / nombre_total_affirmations
```

**Interprétation** :
- ✓ Score élevé (> 0.8) : La réponse ne contient que des informations du contexte
- ⚠ Score moyen (0.6-0.8) : Quelques affirmations non supportées
- ✗ Score faible (< 0.6) : Le LLM invente des informations (hallucinations)

**Exemple** :
- Question : "Où se déroule l'événement X ?"
- Contexte : "L'événement X a lieu à Paris"
- Réponse fidèle : "L'événement X se déroule à Paris"
- Réponse non fidèle : "L'événement X se déroule à Paris au Palais des Congrès" (détail non présent)

### 2. Answer Relevancy (Pertinence de la réponse)

**Définition** : Mesure si la réponse est pertinente par rapport à la question posée.

**Calcul** :
- Génère plusieurs questions à partir de la réponse
- Calcule la similarité cosinus entre ces questions et la question originale

**Interprétation** :
- ✓ Score élevé (> 0.8) : La réponse répond directement à la question
- ⚠ Score moyen (0.6-0.8) : La réponse contient des informations hors-sujet
- ✗ Score faible (< 0.6) : La réponse ne répond pas à la question

**Exemple** :
- Question : "Quand commence l'événement X ?"
- Réponse pertinente : "L'événement X commence le 15 mars 2026 à 19h"
- Réponse peu pertinente : "L'événement X est organisé par la compagnie Y. Il se déroule à Paris. La billetterie ouvre en février." (ne répond pas directement)

### 3. Context Precision (Précision du contexte)

**Définition** : Mesure si les contextes récupérés sont pertinents pour répondre à la question.

**Calcul** :
- Évalue chaque contexte individuellement
- Pénalise les contextes non pertinents, surtout s'ils apparaissent en premier

**Interprétation** :
- ✓ Score élevé (> 0.8) : Les contextes récupérés sont pertinents et bien classés
- ⚠ Score moyen (0.6-0.8) : Quelques contextes non pertinents
- ✗ Score faible (< 0.6) : Beaucoup de bruit dans les contextes

**Impact** :
- Affecte la qualité de la génération (plus de bruit = réponse moins précise)
- Indique un problème d'indexation ou de modèle d'embedding

### 4. Context Recall (Rappel du contexte)

**Définition** : Mesure si le contexte récupéré contient toutes les informations nécessaires pour répondre correctement (nécessite une ground_truth).

**Calcul** :
```
Context Recall = nombre_éléments_gt_dans_contexte / nombre_total_éléments_gt
```

**Interprétation** :
- ✓ Score élevé (> 0.8) : Le contexte contient toutes les informations nécessaires
- ⚠ Score moyen (0.6-0.8) : Quelques informations manquantes
- ✗ Score faible (< 0.6) : Le contexte ne contient pas assez d'informations

**Impact** :
- Affecte la complétude de la réponse
- Indique un problème de chunking ou de top_k trop faible

---

## Jeu de données de test

Le fichier `tests/test_rag_evaluation.py` contient **10 questions de test** réparties en 3 catégories :

### Questions simples (6 questions)

Questions directes nécessitant une recherche simple :
- "Quels sont les événements musicaux prévus en janvier 2026 ?"
- "Où se déroule l'événement 'Lettres non-écrites' ?"
- "Quand commence l'événement X ?"

### Questions complexes (3 questions)

Questions nécessitant une recherche sémantique ou des informations implicites :
- "Y a-t-il des événements liés à la cybersécurité ?"
- "Quels sont les événements culturels gratuits ?"
- "Peux-tu me donner des informations sur les événements avec Erik Satie ?"

### Questions pièges (1 question)

Questions hors-sujet pour tester les limites :
- "Quel temps fera-t-il demain à Paris ?"

### Structure d'une question de test

```python
{
    "question": "La question posée",
    "ground_truth": "La réponse idéale attendue",
    "category": "simple|complexe|piège",
    "description": "Description du cas de test"
}
```

---

## Installation

### Option 1 : Docker (recommandé)

Ragas a été ajouté au `docker/requirements.txt`. Il suffit de reconstruire l'image :

```bash
# Arrêter les conteneurs existants
docker compose -f docker/docker-compose.yml down

# Reconstruire l'image avec Ragas
docker compose -f docker/docker-compose.yml build

# Redémarrer les conteneurs
docker compose -f docker/docker-compose.yml up -d
```

### Option 2 : Installation locale

```bash
pip install ragas datasets
```

**Note** : Ragas nécessite :
- Python >= 3.8
- Un LLM configuré (OpenAI, Anthropic, Mistral, etc.)
- La bibliothèque `datasets` de Hugging Face

---

## Exécution des tests

### Prérequis

1. **L'API doit être en cours d'exécution** :

```bash
docker compose -f docker/docker-compose.yml up -d

# Vérifier que l'API fonctionne
curl http://localhost:8000/health
```

2. **Les documents doivent être indexés** :

```bash
# Si l'index n'existe pas encore
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/build_index.py
```

3. **Un LLM doit être configuré** (pour Ragas) :

Ragas utilise un LLM pour calculer les métriques. Configurez une clé API dans `.env` :

```bash
# OpenAI (recommandé pour Ragas)
OPENAI_API_KEY=sk-your-key-here

# Ou Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Ou Mistral
MISTRAL_API_KEY=your-key-here
```

### Lancer l'évaluation

```bash
# Dans Docker (recommandé)
docker compose -f docker/docker-compose.yml exec rag-system \
  python tests/test_rag_evaluation.py

# Ou localement
python tests/test_rag_evaluation.py
```

### Durée d'exécution

L'évaluation complète prend **5 à 10 minutes** :
- Collecte des réponses : ~1-2 minutes (10 questions)
- Calcul des métriques Ragas : ~3-8 minutes (appels au LLM)

---

## Interprétation des résultats

### Format de sortie

```
======================================================================
RÉSULTATS DE L'ÉVALUATION RAGAS
======================================================================

### Scores globaux (0 à 1, plus élevé = meilleur) ###

Fidélité au contexte         [████████████████████░░░░░░░░░] 0.852
Pertinence de la réponse     [██████████████████████░░░░░░░] 0.789
Précision du contexte        [███████████████████████░░░░░░] 0.823
Rappel du contexte           [████████████████████████░░░░░] 0.845

Score moyen                  : 0.827

### Analyse par catégorie ###

Simple          : 6 question(s)
Complexe        : 3 question(s)
Piège           : 1 question(s)

### Recommandations ###

✓ Excellente performance globale du système RAG !
```

### Barèmes de notation

| Score | Niveau | Signification |
|-------|--------|---------------|
| 0.9 - 1.0 | Excellent | Performance quasi-parfaite |
| 0.8 - 0.9 | Très bon | Quelques améliorations mineures |
| 0.7 - 0.8 | Bon | Performance acceptable |
| 0.6 - 0.7 | Moyen | Améliorations recommandées |
| 0.5 - 0.6 | Faible | Améliorations nécessaires |
| < 0.5 | Très faible | Révision majeure nécessaire |

### Fichiers de sortie

Les résultats sont sauvegardés dans `tests/results/` :

```
tests/results/
├── rag_evaluation_dataset_20260114_143052.json
└── rag_evaluation_results_20260114_143052.json
```

**Dataset JSON** : Contient les questions, réponses, contextes et ground truths
**Results JSON** : Contient les scores de chaque métrique

---

## Amélioration du système

### Si Faithfulness est faible (< 0.7)

**Problème** : Le LLM invente des informations (hallucinations)

**Solutions** :
1. Améliorer le prompt système pour insister sur l'utilisation stricte du contexte
2. Réduire la température du LLM (ex: 0.3 au lieu de 0.7)
3. Ajouter des instructions explicites contre les hallucinations

**Exemple de prompt amélioré** :
```
Réponds uniquement en te basant sur le contexte fourni.
Ne jamais inventer d'informations.
Si tu ne trouves pas la réponse dans le contexte, dis-le explicitement.
```

### Si Answer Relevancy est faible (< 0.7)

**Problème** : Les réponses ne sont pas pertinentes ou contiennent trop d'informations hors-sujet

**Solutions** :
1. Améliorer le prompt pour demander des réponses concises
2. Affiner le modèle d'embedding pour améliorer la pertinence des chunks
3. Ajuster la température du LLM

### Si Context Precision est faible (< 0.7)

**Problème** : Les contextes récupérés ne sont pas pertinents

**Solutions** :
1. **Changer le modèle d'embedding** :
   - Essayer `all-mpnet-base-v2` (plus performant)
   - Ou `multilingual-e5-large` pour du multilingue
2. **Améliorer l'indexation** :
   - Réduire la taille des chunks
   - Améliorer le chunking (utiliser RecursiveCharacterTextSplitter)
3. **Optimiser FAISS** :
   - Ajuster `nprobe` (plus élevé = plus précis mais plus lent)
   - Utiliser un index plus précis (ex: IVFFlat → Flat)

### Si Context Recall est faible (< 0.7)

**Problème** : Le contexte ne contient pas toutes les informations nécessaires

**Solutions** :
1. **Augmenter `top_k`** :
   - Essayer `top_k=10` au lieu de `top_k=5`
2. **Améliorer le chunking** :
   - Augmenter la taille des chunks
   - Augmenter l'overlap entre chunks
3. **Vérifier l'indexation** :
   - S'assurer que tous les documents sont bien indexés
   - Vérifier qu'il n'y a pas de perte d'information lors du parsing

### Workflow d'amélioration

```
1. Exécuter l'évaluation baseline
   ↓
2. Identifier les métriques faibles
   ↓
3. Appliquer les solutions ciblées
   ↓
4. Re-exécuter l'évaluation
   ↓
5. Comparer les résultats
   ↓
6. Itérer jusqu'à satisfaction
```

---

## Exemples de configurations

### Configuration haute précision

```python
# .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
FAISS_INDEX_TYPE=Flat  # Plus précis mais plus lent
TOP_K_RESULTS=10
MAX_CHUNK_SIZE=256
CHUNK_OVERLAP=50
```

**Avantages** : Meilleure précision
**Inconvénients** : Plus lent, nécessite plus de mémoire

### Configuration équilibrée

```python
# .env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_TYPE=IVFFlat
TOP_K_RESULTS=5
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

**Avantages** : Bon équilibre vitesse/précision
**Inconvénients** : Performance moyenne

### Configuration haute vitesse

```python
# .env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_TYPE=IVFFlat
TOP_K_RESULTS=3
MAX_CHUNK_SIZE=1024
CHUNK_OVERLAP=0
```

**Avantages** : Très rapide
**Inconvénients** : Précision réduite

---

## Troubleshooting

### Erreur : "No module named 'ragas'"

**Solution** : Reconstruire l'image Docker ou installer ragas localement
```bash
docker compose -f docker/docker-compose.yml build
```

### Erreur : "API not accessible"

**Solution** : Vérifier que l'API est démarrée
```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs rag-system
```

### Erreur : "Index is empty"

**Solution** : Construire l'index
```bash
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/build_index.py
```

### Erreur : "OpenAI API key not configured"

**Solution** : Ajouter la clé API dans `.env`
```bash
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
docker compose -f docker/docker-compose.yml restart
```

### L'évaluation prend trop de temps

**Solutions** :
- Utiliser un LLM plus rapide (ex: gpt-3.5-turbo)
- Réduire le nombre de questions de test
- Utiliser un modèle local (Ollama)

---

## Références

- [Documentation Ragas](https://docs.ragas.io/)
- [GitHub Ragas](https://github.com/explodinggradients/ragas)
- [Paper RAG Evaluation](https://arxiv.org/abs/2309.15217)
- [Anthropic - Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

## Licence

MIT
