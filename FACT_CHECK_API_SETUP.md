# Setting Up Optional LLM APIs

The fact-checking system works with **Claude (Anthropic) and ChatGPT (OpenAI)** by default. The other three models are optional — you can add them for broader consensus coverage.

## Required: Claude (Anthropic)

```bash
export ANTHROPIC_API_KEY="sk-ant-abc123..."
```

Get your key from: https://console.anthropic.com/account/keys

## Required: ChatGPT (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
```

Get your key from: https://platform.openai.com/account/api-keys

## Optional: DeepSeek

DeepSeek offers a cheap alternative model via OpenAI-compatible endpoint.

```bash
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
```

Get your key from: https://platform.deepseek.com/account/api_keys

**Wire into `fact_check.py` by updating `_initial_fact_check()` around line 161:**

```python
elif model == "deepseek-chat":
    from openai import OpenAI
    ds_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    response = ds_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content
    self.cost_tracker[model] += 0.2
```

## Optional: Gemini (Google)

```bash
pip install google-generativeai
export GEMINI_API_KEY="..."
```

Get your key from: https://aistudio.google.com/app/apikey

**Wire into `fact_check.py`:**

```python
elif model == "gemini-2.0-flash":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    response = gemini_model.generate_content(prompt)
    text = response.text
    self.cost_tracker[model] += 0.15
```

## Optional: Llama (Together AI or Groq)

### Via Together AI

```bash
export TOGETHER_API_KEY="..."
```

Get your key from: https://together.ai/dashboard

**Wire into `fact_check.py`:**

```python
elif model == "llama-3-70b":
    import together
    together.api_key = os.getenv("TOGETHER_API_KEY")
    response = together.Complete.create(
        prompt=prompt,
        model="togethercomputer/llama-3-70b-chat",
        max_tokens=1000,
    )
    text = response["output"]["choices"][0]["text"]
    self.cost_tracker[model] += 0.1
```

### Via Groq

```bash
export GROQ_API_KEY="..."
```

Get your key from: https://console.groq.com

**Wire into `fact_check.py`:**

```python
elif model == "llama-3-70b":
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3-70b-8192",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content
    self.cost_tracker[model] += 0.05
```

## Minimal Setup

If you want to start with the lowest overhead, use:
- **Claude** (Anthropic) — most reliable for fact-checking
- **ChatGPT** (OpenAI) — good coverage, familiar reasoning

This gives you 2 models, enough for the system to function. Consensus requires 3, so you can still get consensus on strong issues (either both agree, or one agrees with any third model you add).

## Cost Comparison (per 1000 tokens)

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| Claude 3.5 Sonnet | $3.00 | $15.00 | Most expensive; very good reasoning |
| GPT-4o mini | $0.15 | $0.60 | Cheap; good for debates |
| DeepSeek | $0.14 | $0.28 | Cheapest; reasonable quality |
| Gemini 2.0 Flash | $0.075 | $0.30 | Very cheap; experimental |
| Llama 3 (Together) | $0.90 | $0.90 | Mid-range; solid reasoning |
| Llama 3 (Groq) | Free | Free | Excellent value; rate limited |

**Recommended budget-conscious setup:**
1. Claude (reasoning power)
2. DeepSeek (cost efficiency)
3. Groq Llama (free backup)

**Cost per article with this setup:** ~$2–4 (under your $10 budget).

## Testing Your Setup

Once you've added a new model, test it locally:

```python
from fact_check import FactCheckOrchestrator

orchestrator = FactCheckOrchestrator()
findings = orchestrator._initial_fact_check("new-model-name", "Test Article", "Some test content...")
print(f"Got {len(findings)} findings from new-model-name")
```

If you get findings without errors, the API is wired correctly.

---

**Don't want to set these up right now?** Start with Claude + GPT. They're the core models. Add others later as needed.
