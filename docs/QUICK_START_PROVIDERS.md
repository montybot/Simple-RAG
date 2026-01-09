# Quick Start: LLM Providers

## 🚀 3-Step Setup for Each Provider

### Option 1: OpenAI (Best Quality)

```bash
# 1. Get API key from https://platform.openai.com/api-keys

# 2. Edit .env
cp .env.example .env
nano .env

# Add this:
OPENAI_API_KEY=sk-your-actual-key-here
MODEL_NAME=gpt-4-turbo-preview

# 3. Done! The system will auto-detect OpenAI
```

---

### Option 2: Anthropic Claude (Best Balance)

```bash
# 1. Get API key from https://console.anthropic.com/

# 2. Edit .env
cp .env.example .env
nano .env

# Add this:
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
MODEL_NAME=claude-3-5-sonnet-20241022

# 3. Done! The system will auto-detect Claude
```

---

### Option 3: Mistral AI (Best Value)

```bash
# 1. Get API key from https://console.mistral.ai/

# 2. Edit .env
cp .env.example .env
nano .env

# Add this:
MISTRAL_API_KEY=your-actual-key-here
MODEL_NAME=mistral-large-latest
# Optional: Use Mistral embeddings too
EMBEDDING_MODEL=mistral-embed

# 3. Done! The system will auto-detect Mistral AI
```

---

### Option 4: Ollama (FREE!)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model
ollama pull llama3.2

# 3. Edit .env
cp .env.example .env
nano .env

# Add this (NO API KEY NEEDED!):
MODEL_NAME=llama3.2

# 4. Done! The system will auto-detect Ollama
```

---

## ✅ Test Your Setup

```bash
# Start the system
docker compose -f docker/docker-compose.yml up -d

# Test a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, test message", "top_k": 3}'

# Check logs to see which provider was detected
docker compose -f docker/docker-compose.yml logs | grep "Using"
```

You should see one of:
- `Using OpenAI: gpt-4-turbo-preview`
- `Using Anthropic Claude: claude-3-5-sonnet-20241022`
- `Using Mistral AI API: mistral-large-latest`
- `Using Ollama local model: llama3.2`

---

## 💡 Quick Tips

### Switch Providers Anytime

Just edit `.env` and change `MODEL_NAME`:

```bash
# Use OpenAI
MODEL_NAME=gpt-4-turbo-preview

# Or use Claude
MODEL_NAME=claude-3-5-sonnet-20241022

# Or use Mistral AI
MODEL_NAME=mistral-large-latest

# Or use local Ollama
MODEL_NAME=llama3.2
```

No code changes needed! The system auto-detects the provider.

### Cost-Free Testing

Use Ollama for unlimited free testing:

```bash
MODEL_NAME=llama3.2
```

Then switch to paid providers for production:

```bash
MODEL_NAME=claude-3-5-sonnet-20241022
```

### Best of Both Worlds

- **Development:** Ollama (free)
- **Budget:** Mistral AI (3-4x cheaper)
- **Production:** Claude or GPT-4 (best quality)

---

## 📋 Provider Comparison

| Provider | Setup Time | Cost | Quality | Best For |
|----------|------------|------|---------|----------|
| **OpenAI** | 2 min | $$$ | ⭐⭐⭐⭐⭐ | Production (best quality) |
| **Claude** | 2 min | $$$ | ⭐⭐⭐⭐⭐ | Production (large context) |
| **Mistral** | 2 min | $$ | ⭐⭐⭐⭐⭐ | Production (best value) |
| **Ollama** | 5 min | FREE | ⭐⭐⭐⭐ | Development |

---

## 🔧 Troubleshooting

### "Import could not be resolved"

Install the provider package:

```bash
# For Anthropic
pip install langchain-anthropic

# For Mistral
pip install langchain-mistralai

# For Ollama
pip install langchain-ollama
```

### "Connection refused"

For Ollama, make sure it's running:

```bash
ollama serve
```

### "Authentication error"

Check your API key in `.env` is correct and has credits.

---

## 📚 More Documentation

- **Full Provider Guide:** [LLM_PROVIDERS.md](LLM_PROVIDERS.md)
- **Mistral AI Detailed Setup:** [MISTRAL_SETUP.md](MISTRAL_SETUP.md)
