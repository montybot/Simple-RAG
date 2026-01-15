# Documentation API - Puls-Events RAG System

## 📚 Documentation interactive

FastAPI génère automatiquement une documentation interactive pour l'API :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI JSON** : http://localhost:8000/openapi.json

Ces interfaces permettent de :
- Visualiser tous les endpoints disponibles
- Tester les requêtes directement depuis le navigateur
- Voir les schémas de données (request/response models)
- Consulter les codes de statut HTTP

---

## 🚀 Vue d'ensemble

L'API REST fournit des endpoints pour :
- Interroger le système RAG (recherche + génération)
- Uploader et indexer de nouveaux documents
- Reconstruire l'index complet
- Consulter les statistiques du système

**Base URL** : `http://localhost:8000`

**Format** : JSON

---

## 📋 Endpoints

### 1. Health Check

Vérifie l'état de santé de l'API et de l'index.

```http
GET /health
```

**Réponse 200 OK :**
```json
{
  "status": "healthy",
  "index_size": 64,
  "stats": {
    "vector_store": {
      "total_vectors": 64,
      "index_type": "IVFFlat",
      "dimension": 1024
    },
    "embedding_model": {
      "model": "mistral-embed",
      "provider": "Mistral AI"
    }
  }
}
```

**Exemple cURL :**
```bash
curl http://localhost:8000/health
```

---

### 2. Query (Interrogation RAG)

Lance une recherche vectorielle puis génère une réponse avec le LLM.

```http
POST /query
```

**Request Body :**
```json
{
  "question": "string",           // Question de l'utilisateur
  "top_k": 5,                      // (optionnel) Nombre de documents à récupérer
  "temperature": 0.7,              // (optionnel) Créativité du LLM (0.0-1.0)
  "max_tokens": 512,               // (optionnel) Longueur max de la réponse
  "top_p": 0.9,                    // (optionnel) Diversité du vocabulaire (0.0-1.0)
  "system_prompt": "string"        // (optionnel) Prompt système pour guider le LLM
}
```

**Réponse 200 OK :**
```json
{
  "answer": "string",              // Réponse générée par le LLM
  "sources": [                     // Documents sources utilisés
    {
      "file": "string",            // Chemin du fichier source
      "title": "string",           // Titre de l'événement
      "score": 0.52,               // Score de similarité (0-1, plus bas = meilleur)
      "excerpt": "string"          // Extrait du document
    }
  ],
  "metadata": {
    "query_time_ms": 245.3,        // Temps de traitement total
    "documents_searched": 5        // Nombre de documents consultés
  }
}
```

**Exemples cURL :**

```bash
# Requête simple
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels concerts ce week-end ?"
  }'

# Requête avec paramètres personnalisés
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Événements japonais à Paris",
    "top_k": 3,
    "temperature": 0.5,
    "max_tokens": 400
  }'

# Requête avec system prompt
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Yokai Matsuri",
    "top_k": 5,
    "temperature": 0.55,
    "max_tokens": 500,
    "top_p": 0.93,
    "system_prompt": "Vous êtes un assistant enthousiaste. Répondez en français."
  }'
```

**Exemple Python :**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "Quels événements en février ?",
        "top_k": 5,
        "temperature": 0.7,
        "max_tokens": 500
    }
)

data = response.json()
print(f"Réponse: {data['answer']}")
print(f"Sources: {len(data['sources'])} documents")
print(f"Temps: {data['metadata']['query_time_ms']}ms")
```

**Codes d'erreur :**
- `500 Internal Server Error` : Erreur lors du traitement de la requête
- `503 Service Unavailable` : API Mistral temporairement surchargée (retry automatique)

---

### 3. Upload Document

Upload et indexe un nouveau document dans le système.

```http
POST /documents/upload
Content-Type: multipart/form-data
```

**Paramètres :**
- `file` : Fichier à uploader (PDF, DOCX, TXT, HTML, CSV)

**Réponse 200 OK :**
```json
{
  "status": "success",
  "document_id": "events_data.csv",
  "message": "Document 'events_data.csv' indexed successfully"
}
```

**Exemple cURL :**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/document.pdf"
```

**Exemple Python :**
```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f}
    )

print(response.json())
```

**Notes :**
- Le document est automatiquement ajouté à l'index existant
- L'index est sauvegardé après l'upload
- Formats supportés : `.pdf`, `.docx`, `.txt`, `.html`, `.csv`

---

### 4. Rebuild Index

Reconstruit l'index complet à partir de tous les documents du répertoire raw.

```http
POST /index/rebuild
```

**Réponse 200 OK :**
```json
{
  "status": "success",
  "message": "Index rebuilt successfully",
  "stats": {
    "vector_store": {
      "total_vectors": 64,
      "index_type": "IVFFlat"
    }
  }
}
```

**Exemple cURL :**
```bash
curl -X POST http://localhost:8000/index/rebuild
```

**Notes :**
- **ATTENTION** : Cette opération supprime l'index existant et le reconstruit entièrement
- Tous les documents du dossier `data/raw/` sont réindexés
- L'opération peut prendre plusieurs minutes selon le volume de documents
- L'index est automatiquement sauvegardé après reconstruction

---

### 5. Get Statistics

Récupère les statistiques détaillées du système.

```http
GET /stats
```

**Réponse 200 OK :**
```json
{
  "vector_store": {
    "total_vectors": 64,
    "index_type": "IVFFlat",
    "dimension": 1024,
    "nlist": 64,
    "nprobe": 10,
    "is_trained": true
  },
  "embedding_model": {
    "model": "mistral-embed",
    "provider": "Mistral AI",
    "dimension": 1024
  },
  "llm": {
    "model": "mistral-large-latest",
    "provider": "Mistral AI"
  },
  "configuration": {
    "chunk_size": 2048,
    "chunk_overlap": 200
  }
}
```

**Exemple cURL :**
```bash
curl http://localhost:8000/stats
```

---

## 🔧 Paramètres LLM

### Temperature (0.0 - 1.0)

Contrôle la **créativité** et le **caractère aléatoire** des réponses.

| Valeur | Comportement | Cas d'usage |
|--------|--------------|-------------|
| `0.0 - 0.3` | Très déterministe, factuel | Questions techniques, informations précises |
| `0.4 - 0.7` | Équilibré (recommandé) | Usage général, conversation naturelle |
| `0.8 - 1.0` | Très créatif, varié | Brainstorming, suggestions créatives |

**Valeur par défaut** : `0.7`
**Valeur recommandée Puls-Events** : `0.55`

### Max Tokens (128 - 2048)

Définit la **longueur maximale** de la réponse en tokens (~0.75 mot par token en français).

| Valeur | Longueur approx. | Cas d'usage |
|--------|------------------|-------------|
| `128-256` | 1-2 paragraphes | Réponses courtes, définitions |
| `256-512` | 2-4 paragraphes | Réponses moyennes (recommandé) |
| `512-1024` | Articles courts | Explications détaillées |
| `1024-2048` | Articles longs | Analyses approfondies |

**Valeur par défaut** : `512`
**Valeur recommandée Puls-Events** : `500`

### Top P (0.0 - 1.0)

Contrôle la **diversité** des mots choisis via *nucleus sampling*.

| Valeur | Comportement | Cas d'usage |
|--------|--------------|-------------|
| `0.1 - 0.5` | Vocabulaire limité, prévisible | Réponses très structurées |
| `0.6 - 0.9` | Diversité équilibrée (recommandé) | Conversations naturelles |
| `0.9 - 1.0` | Maximum de diversité | Textes créatifs |

**Valeur par défaut** : `0.9`
**Valeur recommandée Puls-Events** : `0.93`

### Top K (1 - 20)

Nombre de **documents similaires** à récupérer pour construire le contexte.

| Valeur | Comportement | Cas d'usage |
|--------|--------------|-------------|
| `1-3` | Contexte minimal, précis | Questions très spécifiques |
| `5-7` | Contexte équilibré (recommandé) | Usage général |
| `10-20` | Contexte large | Questions larges, comparaisons |

**Valeur par défaut** : `5`

Voir [docs/LLM_PARAMETERS.md](LLM_PARAMETERS.md) pour plus de détails.

---

## 🎯 System Prompt

Le **system prompt** est un paramètre optionnel mais crucial qui guide le comportement du LLM.

### Pourquoi utiliser un system prompt ?

- **Définit le rôle** : "Vous êtes un assistant virtuel pour Puls-Events"
- **Fixe le ton** : Chaleureux, enthousiaste, professionnel
- **Établit les règles** : Utiliser uniquement les informations du contexte
- **Gère les dates** : Comparaison correcte avec la date actuelle

### ⚠️ Architecture importante

Le system prompt est traité **séparément** de la question :

1. **Vector Search** : Utilise UNIQUEMENT la question brute (pas le system prompt)
2. **LLM Generation** : Utilise system_prompt + contexte + question

```json
{
  "question": "Yokai Matsuri",              // Pour la recherche vectorielle
  "system_prompt": "[Instructions...]"      // Pour la génération LLM
}
```

**Ne jamais** concaténer le system prompt avec la question avant l'envoi !

Voir [docs/SYSTEM_PROMPT_ARCHITECTURE.md](SYSTEM_PROMPT_ARCHITECTURE.md) pour plus de détails.

### Exemple de system prompt

```python
SYSTEM_PROMPT = """
### RÔLE :
Vous êtes l'assistant virtuel de Puls-Events.

### OBJECTIF :
Aider les utilisateurs à découvrir des événements culturels.

### RÈGLES :
- Rester factuel
- Utiliser uniquement les informations du contexte
- Répondre en français avec enthousiasme
"""
```

---

## 📊 Scores de similarité

Les scores retournés dans les sources représentent la **distance L2** entre les embeddings :

| Score | Qualité | Interprétation |
|-------|---------|----------------|
| `0.0 - 0.3` | Excellente | Match exact ou quasi-exact |
| `0.3 - 0.5` | Bonne | Documents très pertinents |
| `0.5 - 0.8` | Moyenne | Documents potentiellement pertinents |
| `0.8 - 1.5` | Faible | Documents peu pertinents |
| `> 1.5` | Très faible | Documents non pertinents |

**Note** : Plus le score est **bas**, meilleure est la correspondance (distance L2).

---

## 🔄 Gestion des erreurs

### Erreur 500 : Internal Server Error

**Causes possibles :**
- Index FAISS non chargé
- Erreur lors de la génération LLM
- Document corrompu lors de l'upload

**Solution :**
- Vérifier les logs : `docker compose -f docker/docker-compose.yml logs rag-system`
- Tester le health check : `curl http://localhost:8000/health`
- Reconstruire l'index si nécessaire : `curl -X POST http://localhost:8000/index/rebuild`

### Erreur 503 : Service Unavailable

**Cause :**
- API Mistral temporairement surchargée

**Solution :**
- Attendre quelques secondes et réessayer
- Le système fait automatiquement des retries avec backoff exponentiel

### Timeout (>120s)

**Cause :**
- API Mistral temporairement indisponible
- Requête trop complexe

**Solution :**
- Réessayer après quelques secondes
- Réduire `max_tokens` pour accélérer la génération
- Réduire `top_k` pour moins de documents à traiter

---

## 🐍 SDK Python

Exemple d'utilisation avec la librairie `requests` :

```python
import requests
from typing import Dict, Optional

class PulsEventsAPI:
    """Client Python pour l'API Puls-Events RAG."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health(self) -> Dict:
        """Vérifie l'état de santé de l'API."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Interroge le système RAG.

        Args:
            question: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            temperature: Créativité du LLM (0.0-1.0)
            max_tokens: Longueur max de la réponse
            top_p: Diversité du vocabulaire (0.0-1.0)
            system_prompt: Prompt système optionnel

        Returns:
            Dict avec answer, sources, et metadata
        """
        payload = {
            "question": question,
            "top_k": top_k,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    def upload_document(self, file_path: str) -> Dict:
        """Upload et indexe un nouveau document."""
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files={"file": f}
            )
        response.raise_for_status()
        return response.json()

    def rebuild_index(self) -> Dict:
        """Reconstruit l'index complet."""
        response = requests.post(f"{self.base_url}/index/rebuild")
        response.raise_for_status()
        return response.json()

    def stats(self) -> Dict:
        """Récupère les statistiques du système."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()


# Utilisation
api = PulsEventsAPI()

# Health check
health = api.health()
print(f"Index size: {health['index_size']} documents")

# Requête
result = api.query(
    question="Quels événements japonais ce mois-ci ?",
    top_k=5,
    temperature=0.55,
    max_tokens=500,
    system_prompt="Vous êtes un assistant enthousiaste pour Puls-Events."
)

print(f"Réponse: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
print(f"Temps: {result['metadata']['query_time_ms']}ms")

for i, source in enumerate(result['sources'], 1):
    print(f"\nSource {i}: {source['title']}")
    print(f"  Score: {source['score']:.4f}")
    print(f"  Extrait: {source['excerpt'][:100]}...")
```

---

## 🧪 Tests

### Test rapide avec cURL

```bash
# 1. Vérifier que l'API fonctionne
curl http://localhost:8000/health

# 2. Tester une requête simple
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements ce week-end ?"}'

# 3. Consulter les statistiques
curl http://localhost:8000/stats
```

### Test avec Python

```python
import requests

# Test 1: Health check
health = requests.get("http://localhost:8000/health").json()
assert health["status"] == "healthy"
print(f"✅ Index size: {health['index_size']}")

# Test 2: Query
result = requests.post(
    "http://localhost:8000/query",
    json={"question": "Yokai Matsuri"}
).json()

assert "answer" in result
assert len(result["sources"]) > 0
print(f"✅ Query returned {len(result['sources'])} sources")
print(f"✅ Best match score: {result['sources'][0]['score']:.4f}")
```

---

## 📖 Voir aussi

- [LLM_PARAMETERS.md](LLM_PARAMETERS.md) - Guide détaillé des paramètres LLM
- [SYSTEM_PROMPT_ARCHITECTURE.md](SYSTEM_PROMPT_ARCHITECTURE.md) - Architecture system prompt
- [CSV_PROCESSING.md](CSV_PROCESSING.md) - Traitement des fichiers CSV
- [LLM_PROVIDERS.md](LLM_PROVIDERS.md) - Configuration des fournisseurs LLM

---

**Dernière mise à jour** : 2026-01-13
**Version API** : 1.0.0
**Contact** : Support Puls-Events
