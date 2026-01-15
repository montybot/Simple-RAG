# Guide des paramètres LLM

Ce guide explique comment les paramètres du modèle de langage affectent les réponses du système RAG.

## 📊 Paramètres disponibles

### 1. **Temperature** (0.0 - 1.0)

Contrôle la **créativité** et le **caractère aléatoire** des réponses.

| Valeur | Comportement | Cas d'usage |
|--------|-------------|-------------|
| **0.0 - 0.3** | Très déterministe, factuel | Questions techniques, informations précises |
| **0.4 - 0.7** | Équilibré (recommandé) | Usage général, conversation naturelle |
| **0.8 - 1.0** | Très créatif, varié | Brainstorming, suggestions créatives |

**Exemple pratique :**
```
Question : "Quels concerts jazz sont disponibles ?"

Temperature = 0.2 → "Il y a 3 concerts de jazz cette semaine : [liste stricte]"
Temperature = 0.7 → "Plusieurs concerts de jazz sympas cette semaine ! Notamment..."
Temperature = 1.0 → "Oh, excellente question ! Le jazz fleurit cette semaine avec..."
```

**Valeur recommandée pour Puls-Events :** `0.55` (selon le system prompt)

---

### 2. **Max Tokens** (128 - 2048)

Définit la **longueur maximale** de la réponse en tokens (~0.75 mot par token en français).

| Valeur | Longueur approx. | Cas d'usage |
|--------|------------------|-------------|
| **128-256** | 1-2 paragraphes | Réponses courtes, définitions |
| **256-512** | 2-4 paragraphes | Réponses moyennes (recommandé) |
| **512-1024** | Articles courts | Explications détaillées |
| **1024-2048** | Articles longs | Analyses approfondies |

**Exemple pratique :**
```
Question : "Décris l'exposition au musée"

Max Tokens = 100 → Réponse concise (3-4 phrases)
Max Tokens = 500 → Réponse détaillée avec contexte (recommandé)
Max Tokens = 1000 → Réponse très complète avec tous les détails
```

**Valeur recommandée pour Puls-Events :** `500` (selon le system prompt)

---

### 3. **Top P** (0.0 - 1.0)

Contrôle la **diversité** des mots choisis via *nucleus sampling*.

| Valeur | Comportement | Cas d'usage |
|--------|-------------|-------------|
| **0.1 - 0.5** | Vocabulaire limité, prévisible | Réponses très structurées |
| **0.6 - 0.9** | Diversité équilibrée (recommandé) | Conversations naturelles |
| **0.9 - 1.0** | Maximum de diversité | Textes créatifs |

**Comment ça marche :**
- `Top P = 0.9` signifie que le modèle choisit parmi les mots représentant 90% de la probabilité cumulée
- Plus élevé = plus de choix possibles = réponses plus variées

**Exemple pratique :**
```
Question : "Quels événements ce week-end ?"

Top P = 0.5 → "Voici les événements de ce week-end : [format strict]"
Top P = 0.93 → "Ce week-end, découvrez plusieurs événements passionnants..." (recommandé)
Top P = 1.0 → Utilise tout le vocabulaire disponible, très varié
```

**Valeur recommandée pour Puls-Events :** `0.93` (selon le system prompt)

---

## 🎯 Configurations recommandées

### Configuration par défaut (Puls-Events)
```json
{
  "temperature": 0.55,
  "max_tokens": 500,
  "top_p": 0.93
}
```
**Usage :** Conversations naturelles, enthousiasme équilibré, informations complètes

---

### Configuration factuelle
```json
{
  "temperature": 0.2,
  "max_tokens": 300,
  "top_p": 0.7
}
```
**Usage :** Réponses très précises, horaires exacts, informations techniques

---

### Configuration créative
```json
{
  "temperature": 0.8,
  "max_tokens": 800,
  "top_p": 0.95
}
```
**Usage :** Suggestions personnalisées, descriptions enthousiastes, recommandations

---

### Configuration concise
```json
{
  "temperature": 0.3,
  "max_tokens": 150,
  "top_p": 0.8
}
```
**Usage :** Réponses ultra-courtes, listes simples, informations rapides

---

## 🔄 Compatibilité multi-modèles

Ces paramètres sont compatibles avec tous les modèles supportés :

| Fournisseur | Modèles | Paramètres supportés |
|-------------|---------|----------------------|
| **Mistral AI** | mistral-small, mistral-medium, mistral-large | ✅ Tous |
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-3.5-turbo | ✅ Tous |
| **Anthropic** | claude-3-opus, claude-3-sonnet | ✅ Tous |
| **Ollama** | llama3.2, mistral (local) | ✅ Tous (num_predict pour max_tokens) |

**Note :** Le système adapte automatiquement les paramètres selon le fournisseur.

---

## 💡 Conseils d'optimisation

### Pour économiser les tokens
- Réduire `max_tokens` à 300-400
- Utiliser `temperature` bas (0.2-0.4)
- Garder `top_p` modéré (0.7-0.8)

### Pour des réponses plus naturelles
- Augmenter légèrement `temperature` (0.6-0.7)
- Utiliser `top_p` élevé (0.9-0.95)
- Donner plus d'espace avec `max_tokens` (500-800)

### Pour du debug/test
- `temperature = 0.0` pour des réponses reproductibles
- `max_tokens` faible pour itérer rapidement
- `top_p = 0.5` pour réduire la variabilité

---

## 📈 Impact sur les coûts

**Coût = nombre de tokens × prix par token**

| Paramètre | Impact sur coût | Explication |
|-----------|----------------|-------------|
| **temperature** | ❌ Aucun | Ne change pas le nombre de tokens |
| **max_tokens** | ✅✅✅ Élevé | Limite directement les tokens générés |
| **top_p** | ❌ Aucun | Ne change pas le nombre de tokens |

**Astuce :** Pour réduire les coûts, ajustez prioritairement `max_tokens`.

---

## 🧪 Expérimentation

Utilisez l'interface Streamlit pour tester différentes combinaisons :

1. Ouvrez http://localhost:8001
2. Ajustez les sliders dans la sidebar
3. Posez la même question avec différents paramètres
4. Comparez les résultats

**Questions de test recommandées :**
- "Quels événements ce week-end à Rouen ?"
- "Décris l'exposition au musée des Beaux-Arts"
- "Recommande-moi un concert de jazz"

---

## 🎭 Paramètres Puls-Events

Les valeurs par défaut de l'interface Streamlit sont optimisées pour Puls-Events :

```python
temperature = 0.55   # Équilibre entre factuel et enthousiaste
max_tokens = 500     # Réponses détaillées mais concises
top_p = 0.93         # Conversation naturelle et fluide
```

Ces valeurs ont été choisies selon le system prompt pour offrir :
- ✅ Informations factuelles et exactes
- ✅ Ton chaleureux et enthousiaste
- ✅ Réponses complètes sans verbosité excessive
- ✅ Conversation naturelle et engageante

---

**Dernière mise à jour :** 2026-01-13
