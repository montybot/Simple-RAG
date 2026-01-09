# LLM Provider Configuration Guide

The RAG system now supports **multiple LLM providers** out of the box. Simply change your `.env` file to switch between providers - no code changes needed!

## Supported Providers

| Provider | Cost | Speed | Quality | Setup Difficulty |
|----------|------|-------|---------|------------------|
| **OpenAI** | $$$ | Fast | Excellent | Easy (API key) |
| **Anthropic Claude** | $$$ | Fast | Excellent | Easy (API key) |
| **Mistral AI** | $$ | Fast | Excellent | Easy (API key) |
| **Ollama (Local)** | Free | Medium | Good | Medium (install) |

---

## 1. OpenAI (Default)

### Setup

1. Get API key from https://platform.openai.com/api-keys

2. Update `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4-turbo-preview
```

### Available Models

- `gpt-4-turbo-preview` - Best quality, higher cost
- `gpt-4` - Excellent quality
- `gpt-3.5-turbo` - Fast, lower cost

### Example

```bash
# .env
OPENAI_API_KEY=sk-proj-abc123...
MODEL_NAME=gpt-4-turbo-preview
```

---

## 2. Anthropic Claude

### Setup

1. Get API key from https://console.anthropic.com/

2. Update `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
MODEL_NAME=claude-3-5-sonnet-20241022
```

### Available Models

- `claude-3-5-sonnet-20241022` - Latest, best balance
- `claude-3-opus-20240229` - Highest capability
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-haiku-20240307` - Fastest, cheapest

### Example

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-xyz789...
MODEL_NAME=claude-3-5-sonnet-20241022
```

### Benefits

- Excellent reasoning
- Large context window (200K tokens)
- Strong at following instructions

---

## 3. Mistral AI

### Setup

1. Get API key from https://console.mistral.ai/

2. Update `.env`:
```bash
MISTRAL_API_KEY=your-mistral-key-here
MODEL_NAME=mistral-large-latest
```

### Available Models

**Chat Models:**
- `mistral-large-latest` - Most capable, best for complex tasks
- `mistral-medium-latest` - Balanced performance
- `mistral-small-latest` - Fast and efficient
- `open-mistral-7b` - Open source, lightweight

**Embeddings:**
- `mistral-embed` - High-quality embeddings (1024 dimensions)

### Example - LLM

```bash
# .env
MISTRAL_API_KEY=abc123xyz...
MODEL_NAME=mistral-large-latest
```

### Example - Embeddings

```bash
# .env
MISTRAL_API_KEY=abc123xyz...
EMBEDDING_MODEL=mistral-embed
```

### Benefits

- Cost-effective (cheaper than OpenAI/Claude)
- Fast inference speed
- European data sovereignty
- Both LLM and embeddings from same provider
- Open source models available

### Pricing (as of 2024)

- **mistral-large:** $4 / 1M input tokens, $12 / 1M output tokens
- **mistral-small:** $1 / 1M input tokens, $3 / 1M output tokens
- **mistral-embed:** $0.10 / 1M tokens

---

## 4. Ollama (Local - FREE!)

### Setup

#### Step 1: Install Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

#### Step 2: Pull a model

```bash
# Pull llama3.2 (4.1GB)
ollama pull llama3.2

# Or other models:
ollama pull mistral      # 4.1GB
ollama pull llama2       # 3.8GB
ollama pull codellama    # 3.8GB
ollama pull phi          # 1.6GB (smaller, faster)
```

#### Step 3: Start Ollama (if not auto-started)

```bash
ollama serve
```

#### Step 4: Update `.env`

```bash
# No API key needed!
MODEL_NAME=llama3.2
```

### Available Models

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2` | 4.1GB | Best overall |
| `mistral` | 4.1GB | Fast, efficient |
| `phi` | 1.6GB | Lightweight |
| `codellama` | 3.8GB | Code-focused |

### Example

```bash
# .env
MODEL_NAME=llama3.2

# That's it! No API key needed
```

### Benefits

- 100% free
- No API costs
- Data stays local
- No rate limits

### Drawbacks

- Slower than cloud providers
- Requires local compute (4-8GB RAM)
- Quality slightly lower than GPT-4/Claude

---

## Quick Switching Guide

### From OpenAI to Claude

```bash
# Comment out OpenAI
# OPENAI_API_KEY=sk-...

# Add Claude
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-5-sonnet-20241022
```

### From OpenAI to Ollama

```bash
# Comment out OpenAI
# OPENAI_API_KEY=sk-...

# Use local model
MODEL_NAME=llama3.2
```

### From Claude to OpenAI

```bash
# Comment out Claude
# ANTHROPIC_API_KEY=sk-ant-...

# Use OpenAI
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4-turbo-preview
```

---

## How Detection Works

The system automatically detects which provider to use based on the `MODEL_NAME`:

```python
# In src/rag_pipeline.py
def _get_llm(self):
    model_lower = self.llm_model.lower()

    if "claude" in model_lower:
        return ChatAnthropic(...)  # Anthropic

    elif "llama" in model_lower or "mistral" in model_lower:
        return ChatOllama(...)  # Ollama local

    else:
        return ChatOpenAI(...)  # OpenAI default
```

---

## Testing Your Configuration

### 1. Check API Connection

```bash
# Test with a simple query
python scripts/query.py "Hello, test message" --verbose
```

### 2. Check Logs

```bash
# Look for provider detection
tail -f logs/rag_*.log

# Should see:
# "Using OpenAI: gpt-4-turbo-preview" or
# "Using Anthropic Claude: claude-3-5-sonnet-20241022" or
# "Using Ollama local model: llama3.2"
```

### 3. Test via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Test query", "top_k": 3}'
```

---

## Troubleshooting

### Error: "Import langchain_anthropic could not be resolved"

**Solution:**
```bash
pip install langchain-anthropic
```

### Error: "Import langchain_ollama could not be resolved"

**Solution:**
```bash
pip install langchain-ollama
```

### Error: "Connection refused to localhost:11434"

**Solution:** Start Ollama service
```bash
ollama serve
```

### Error: "Model not found" (Ollama)

**Solution:** Pull the model first
```bash
ollama pull llama3.2
```

### Error: "AuthenticationError" (OpenAI/Claude)

**Solution:** Check your API key is correct and has credits

---

## Cost Comparison

### OpenAI GPT-4 Turbo
- Input: $10 / 1M tokens
- Output: $30 / 1M tokens
- **Example:** 1000 queries ≈ $2-5

### Anthropic Claude 3.5 Sonnet
- Input: $3 / 1M tokens
- Output: $15 / 1M tokens
- **Example:** 1000 queries ≈ $1-3

### Ollama (Local)
- **Cost:** $0 (free)
- **Trade-off:** Requires local compute

---

## Recommended Setup

### Development/Testing
Use **Ollama** (free, unlimited testing)

```bash
MODEL_NAME=llama3.2
```

### Production
Use **Claude 3.5 Sonnet** (best balance of cost/quality)

```bash
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-5-sonnet-20241022
```

### High-Quality Tasks
Use **GPT-4 Turbo** (best quality)

```bash
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4-turbo-preview
```

---

## Advanced: Custom Provider

Want to add a custom provider? Edit `src/rag_pipeline.py`:

```python
def _get_llm(self):
    model_lower = self.llm_model.lower()

    # Add your custom provider
    if "my-custom-model" in model_lower:
        from langchain_custom import ChatCustom
        return ChatCustom(model=self.llm_model, temperature=0)

    # ... existing providers
```

---

## Quick Reference

```bash
# OpenAI
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4-turbo-preview

# Claude
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-5-sonnet-20241022

# Ollama
MODEL_NAME=llama3.2
```

**That's it!** The system handles everything else automatically.
