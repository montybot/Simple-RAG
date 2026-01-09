# Mistral AI Setup Guide

Complete guide for using Mistral AI with the RAG system - for both LLM and embeddings.

## Why Mistral AI?

✅ **Cost-effective** - 3-4x cheaper than OpenAI/Claude
✅ **Fast** - Low latency responses
✅ **European** - EU data sovereignty
✅ **Unified** - Same provider for LLM + embeddings
✅ **Open Source** - Some models are open source

---

## Quick Setup (3 Steps)

### 1. Get API Key

Visit: https://console.mistral.ai/

1. Sign up for a Mistral account
2. Navigate to "API Keys"
3. Create a new API key
4. Copy the key (starts with your account identifier)

### 2. Configure `.env`

```bash
cp .env.example .env
nano .env
```

**For LLM only:**
```bash
MISTRAL_API_KEY=your-mistral-key-here
MODEL_NAME=mistral-large-latest
```

**For LLM + Embeddings:**
```bash
MISTRAL_API_KEY=your-mistral-key-here
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=mistral-embed
```

### 3. Done!

The system auto-detects Mistral AI from the model names. No code changes needed!

---

## Model Selection

### Chat Models

| Model | Use Case | Cost | Speed |
|-------|----------|------|-------|
| `mistral-large-latest` | Complex reasoning, best quality | $$ | Fast |
| `mistral-medium-latest` | Balanced tasks | $ | Faster |
| `mistral-small-latest` | Simple tasks, high volume | $ | Fastest |
| `open-mistral-7b` | Open source, local fallback | Free* | Fast |

*Open models can also run on Ollama locally

### Embedding Model

| Model | Dimension | Use Case | Cost |
|-------|-----------|----------|------|
| `mistral-embed` | 1024 | Semantic search, RAG | $0.10/1M tokens |

---

## Detection Logic

The system detects Mistral AI based on model naming:

**For LLM:**
- If model name contains `mistral-` (with dash) → Mistral AI API
- If model name contains `mistral` (no dash) → Ollama local

**Examples:**
- `mistral-large` → Mistral AI API ✓
- `mistral-small-latest` → Mistral AI API ✓
- `mistral` → Ollama local ✓
- `llama3.2` → Ollama local ✓

**For Embeddings:**
- If model name contains `mistral` + `embed` → Mistral AI API
- Otherwise → sentence-transformers local

**Examples:**
- `mistral-embed` → Mistral AI API ✓
- `sentence-transformers/all-MiniLM-L6-v2` → Local ✓

---

## Configuration Examples

### Example 1: Mistral LLM + Local Embeddings (Recommended for Testing)

```bash
# .env
MISTRAL_API_KEY=your-key-here
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Free, local
```

**Pros:**
- Free embeddings
- Fast LLM responses
- Cost-effective

### Example 2: Full Mistral Stack (Recommended for Production)

```bash
# .env
MISTRAL_API_KEY=your-key-here
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=mistral-embed
```

**Pros:**
- Single provider for everything
- Consistent quality
- Simplified billing

### Example 3: Budget-Friendly

```bash
# .env
MISTRAL_API_KEY=your-key-here
MODEL_NAME=mistral-small-latest  # Cheaper
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Free
```

**Pros:**
- Lowest cost
- Still good quality
- Fast responses

---

## Pricing Comparison

### LLM Costs (per 1M tokens)

| Provider | Input | Output | Example Cost* |
|----------|-------|--------|---------------|
| **Mistral Large** | $4 | $12 | $0.80 |
| **Mistral Small** | $1 | $3 | $0.20 |
| OpenAI GPT-4 | $10 | $30 | $2.00 |
| Claude Sonnet | $3 | $15 | $0.90 |

*Based on 1000 RAG queries (50k input + 20k output tokens)

### Embedding Costs (per 1M tokens)

| Provider | Cost | Example Cost* |
|----------|------|---------------|
| **Mistral Embed** | $0.10 | $0.50 |
| OpenAI Ada-002 | $0.10 | $0.50 |
| Local (sentence-transformers) | Free | $0 |

*Based on indexing 5M tokens (typical medium corpus)

---

## Testing Your Setup

### 1. Test API Connection

```bash
# Quick test via CLI
python scripts/query.py "Test question" --verbose
```

Look for:
```
Using Mistral AI API: mistral-large-latest
```

### 2. Test Embeddings

```bash
# Build index with Mistral embeddings
python scripts/build_index.py --embedding-model mistral-embed
```

Look for:
```
Using Mistral AI embeddings API
Dimension: 1024
```

### 3. Test via API

```bash
# Start the server
docker compose -f docker/docker-compose.yml up -d

# Query
curl -X POST http://localhost:YOUR_DEFINE_ENV_PORT/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Test Mistral AI", "top_k": 3}'
```

### 4. Check Logs

```bash
# View logs
docker compose -f docker/docker-compose.yml logs | grep -i mistral
```

Should see:
```
Using Mistral AI API: mistral-large-latest
Using Mistral AI embeddings API
```

---

## Troubleshooting

### Error: "Import langchain_mistralai could not be resolved"

**Solution:**
```bash
pip install langchain-mistralai
```

Or rebuild Docker:
```bash
docker compose -f docker/docker-compose.yml build
```

### Error: "Authentication failed"

**Causes:**
- Invalid API key
- No credits in account
- Wrong key format

**Solution:**
1. Check your API key at https://console.mistral.ai/
2. Verify you have credits
3. Ensure key is correctly pasted in `.env`

### Warning: Using Ollama instead of Mistral API

**Cause:** Model name doesn't match Mistral AI pattern

**Solution:**
```bash
# Wrong (will use Ollama):
MODEL_NAME=mistral

# Correct (will use Mistral AI):
MODEL_NAME=mistral-large-latest
```

### Embeddings Not Using Mistral

**Cause:** Wrong embedding model name

**Solution:**
```bash
# Wrong:
EMBEDDING_MODEL=mistral

# Correct:
EMBEDDING_MODEL=mistral-embed
```

---

## Performance Tips

### 1. Choose Right Model for Task

```bash
# Complex reasoning → Large
MODEL_NAME=mistral-large-latest

# General queries → Medium
MODEL_NAME=mistral-medium-latest

# High volume, simple → Small
MODEL_NAME=mistral-small-latest
```

### 2. Batch Embeddings

The system automatically batches embeddings during indexing. For custom code:

```python
from src.embeddings import EmbeddingModel

model = EmbeddingModel("mistral-embed")
# Batch encode for efficiency
embeddings = model.encode_batch(texts, batch_size=32)
```

### 3. Cache Common Queries

Consider implementing query caching for frequent questions to reduce API costs.

---

## Migration Guide

### From OpenAI to Mistral

```bash
# Before (OpenAI)
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-ada-002

# After (Mistral)
MISTRAL_API_KEY=your-key...
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=mistral-embed
```

**Note:** Rebuild index with new embeddings:
```bash
python scripts/build_index.py --force-rebuild
```

### From Claude to Mistral

```bash
# Before (Claude)
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-5-sonnet-20241022

# After (Mistral)
MISTRAL_API_KEY=your-key...
MODEL_NAME=mistral-large-latest
```

### From Ollama to Mistral

```bash
# Before (Ollama - Free)
MODEL_NAME=llama3.2  # No API key

# After (Mistral - Paid)
MISTRAL_API_KEY=your-key...
MODEL_NAME=mistral-large-latest
```

**Why switch?**
- Faster responses
- Better quality
- No local compute needed
- Consistent availability

---

## Best Practices

### 1. Development

Use **Ollama** (free) for development:
```bash
MODEL_NAME=llama3.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 2. Staging

Use **Mistral Small** (cheap) for testing:
```bash
MISTRAL_API_KEY=your-key
MODEL_NAME=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
```

### 3. Production

Use **Mistral Large** (best) for production:
```bash
MISTRAL_API_KEY=your-key
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=mistral-embed
```

---

## FAQ

**Q: Can I use Mistral LLM with OpenAI embeddings?**
A: Yes! Mix and match:
```bash
MISTRAL_API_KEY=your-key
OPENAI_API_KEY=your-key
MODEL_NAME=mistral-large-latest
EMBEDDING_MODEL=text-embedding-ada-002
```

**Q: Do I need to rebuild the index when switching LLMs?**
A: No, only when changing embedding models.

**Q: Can I use Mistral Open Source models?**
A: Yes, via Ollama:
```bash
# Pull the model
ollama pull mistral

# Use it (no API key needed)
MODEL_NAME=mistral
```

**Q: How do embeddings affect costs?**
A: Embeddings are paid once during indexing. LLM costs are per query.

**Q: Is my data sent to Mistral?**
A: Yes, for API-based models. Use Ollama for full local operation.

---

## Resources

- **Mistral Console:** https://console.mistral.ai/
- **Mistral Docs:** https://docs.mistral.ai/
- **Pricing:** https://mistral.ai/pricing/
- **Model List:** https://docs.mistral.ai/getting-started/models/

---

## Support

For issues:
1. Check this guide
2. Review [docs/LLM_PROVIDERS.md](LLM_PROVIDERS.md)
3. Check Mistral documentation
4. Open a GitHub issue

**Happy building with Mistral AI! 🚀**
