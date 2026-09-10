# Multi-LLM Fact-Checking Orchestrator

## System Overview

The fact-checking system runs a structured debate across 5 LLMs (Claude, ChatGPT, DeepSeek, Gemini, Llama) to validate blog articles. The goal: surface only important factual issues, achieve consensus among models, and flag persistent disagreements.

## Core Design

### Consensus Model

**Consensus = 3+ models agreeing on the same issue with consistent severity rating.**

If 3+ models flag the same claim with the same severity (e.g., all rate it severity 6), that's consensus. The issue surfaces in the output.

If models disagree on severity (e.g., one says 3, another says 7), the system marks it as a blocker and continues debate.

### Blocker Detection

**Blocker = a claim where no consensus after 5 rounds of debate per model.**

Each claim can be debated multiple times. If the same LLM has spoken on a claim 5 times and there's still no agreement with the others, the debate is futile and the claim moves to the blockers list.

A blocker shows:
- The claim in question
- Each model's positions across all 5 rounds
- How many rounds were debated

### Severity Scoring

1-10 scale, where:
- 1-2: Trivial (typo, awkward phrasing, minor imprecision)
- 3-4: Noticeable (could confuse some readers, but doesn't change core understanding)
- 5-6: Significant (misleading or wrong in a way that matters)
- 7-8: Major (contradicts established science or introduces false concepts)
- 9-10: Critical (actively dangerous, damages credibility, or completely false)

**Pass threshold: no action items with severity > 2.** This means the article passes fact-check if only trivial issues surface.

### Debate Loop

**Round 1 (Parallel):**
Each LLM independently scans the article and identifies 3-5 important factual issues. This happens in parallel (all 5 models scan simultaneously).

**Rounds 2-100 (Sequential Debate):**
Group issues into "claimed items" (e.g., "the softmax function always outputs probabilities summing to 1").

For each blocker item (unresolved disagreement), ask each model: "Given that other models have taken these positions, do you maintain yours or agree with any of them?"

Loop until:
- All issues reach consensus (all disagreeing models align), OR
- All blockers hit the 5-round threshold, OR
- 100 rounds elapsed

### Output Format

**Report structure:**
```
{
  "article": "Article Title",
  "excerpt": "Short excerpt",
  "status": "PASS" or "NEEDS_REVISION",
  "agreed_upon_changes": [
    {
      "claim": "The specific claim",
      "issue": "What's wrong with it",
      "severity": 7,
      "suggested_fix": "How to fix it",
      "consensus_models": ["claude", "gpt-4o-mini", "gemini"]
    }
  ],
  "consensus_blockers": [
    {
      "claim": "Blocked claim",
      "conflicting_positions": {
        "claude": ["Position A", "Position A", "Position A"],
        "gpt-4o-mini": ["Position B", "Position B", "Position B"]
      },
      "rounds_debated": 5
    }
  ],
  "cost_estimate": 3.25
}
```

**Key fields:**
- `status`: PASS if no action items > severity 2; NEEDS_REVISION otherwise
- `agreed_upon_changes`: ranked by severity, descending
- `consensus_blockers`: items with no resolution after max debate
- `cost_estimate`: total API spend in USD

## Cost Management

**Budget: $10 per article**

Cost assumptions:
- Claude Sonnet: ~$0.50 per fact-check round, $0.30 per debate round
- GPT-4o mini: ~$0.30 per fact-check round, $0.15 per debate round
- DeepSeek/Gemini/Llama: ~$0.20 per round (cheaper APIs)

**Real costs depend on:**
- Article length (we use first 3000 chars)
- Number of debate rounds needed
- Response lengths from each model

The system stops adding new debate rounds if estimated cost exceeds $10.

## Noise Filtering

The system only surfaces "important items" — no trivial nitpicks.

**Filtering rules:**
- No typos unless they confuse meaning (e.g., "image" vs. "image processing")
- No awkward phrasing unless it misleads
- No minor oversimplifications that are pedagogically reasonable (e.g., "pooling reduces size" is fine even though the math is more subtle)
- No disagreement on language style or tone

**Severity > 2 filter applies at output time** — all debate happens on all claimed issues, but the report only shows action items with severity > 2.

## Model Coverage

**Included:**
- Claude 3.5 Sonnet (Anthropic)
- GPT-4o mini (OpenAI)
- DeepSeek Chat (via OpenAI-compatible endpoint)
- Gemini 2.0 Flash (Google)
- Llama (via Together AI or similar)

**Rationale:**
- Broad generalist coverage (Claude, GPT are benchmarks)
- Cost-efficient alternatives (DeepSeek, Llama)
- Specialized vision reasoning (Gemini)

Consensus requires 3+ of these 5.

## Implementation Details

### Data Flow

1. **Extract:** Parse article frontmatter (title, excerpt) and body from MDX
2. **Scan (Round 1):** Send full article to all 5 models in parallel
3. **Aggregate:** Merge findings into unique claims; group by claim ID
4. **Analyze:** Detect consensus vs. blockers
5. **Debate:** For each blocker, query each model with other positions
6. **Loop:** Back to step 4 until consensus or round 100
7. **Report:** Generate JSON with agreed changes + blockers

### State Tracking

```python
findings_by_claim = {
  "claim text": {
    "claude": [Finding(...), Finding(...), Finding(...)],  # one per round
    "gpt-4o-mini": [Finding(...), Finding(...)]
  }
}
claim_round_counts = {
  "claim text": {
    "claude": 3,        # claude has weighed in 3 times on this claim
    "gpt-4o-mini": 2
  }
}
```

Per-claim round counts let us detect when a model has hit the 5-round blocker threshold.

## Cost Tracking

Simple counter per model:
```python
cost_tracker = {
  "claude-3-5-sonnet-20241022": 3.50,
  "gpt-4o-mini": 1.20,
  ...
}
```

Before each debate round, check: `sum(cost_tracker.values()) < 10.0`. If we've hit the budget, stop debating and report blockers as-is.

## API Setup

**Required environment variables:**
```
ANTHROPIC_API_KEY=...      # for Claude
OPENAI_API_KEY=...         # for ChatGPT
DEEPSEEK_API_KEY=...       # for DeepSeek (if using direct endpoint)
GEMINI_API_KEY=...         # for Gemini
LLAMA_API_KEY=...          # for Llama (Together, Groq, etc.)
```

Or use OpenAI-compatible endpoints for some (DeepSeek supports this).

## Running a Fact-Check

```bash
python fact_check.py src/content/posts/paper-1-part-8-neurons/index.mdx
```

Output:
- Console: real-time progress (round numbers, finding counts, consensus status)
- File: `src/content/posts/paper-1-part-8-neurons/index_fact_check.json`

The JSON report is the source of truth for the article's status.

## Limitations & Trade-offs

**By design:**
- First 3000 chars of article (not full length) — limits cost, may miss issues in later sections
- Debate halts at 100 rounds — prevents infinite loops on intractable disagreements
- Blocker threshold at 5 rounds per model — tuned for 30-minute runtime per article
- Severity scoring is model-specific — different LLMs may rate the same issue differently (that's OK; we only count consensus when ratings match)

**Known issues (future work):**
- No deduplication across models (if Claude and GPT both flag the same issue slightly differently, they're treated as separate claims)
- No cost breakdown by model (only total)
- DeepSeek/Gemini/Llama APIs not yet fully wired (placeholders in place)

## Example Output

Article: "Paper #1, Part 8: Neurons and Networks"

```json
{
  "article": "Paper #1, Part 8: Neurons and Networks...",
  "status": "NEEDS_REVISION",
  "agreed_upon_changes": [
    {
      "claim": "The ReLU activation function sets negative values to zero",
      "issue": "This is imprecise. ReLU is max(0, x), so it *outputs* zero for negative inputs, but doesn't literally 'set' them. The distinction matters for understanding backprop through ReLU.",
      "severity": 4,
      "suggested_fix": "Change to: 'The ReLU activation function outputs zero for any negative input, and passes positive values through unchanged.'",
      "consensus_models": ["claude", "gpt-4o-mini", "gemini"]
    },
    {
      "claim": "Stacking layers plus non-linearity lets us learn non-linear patterns",
      "issue": "Unclear what 'stacking layers plus non-linearity' means. Do you mean multiple layers each with activation? Or just activation functions? The phrasing conflates structure and individual component properties.",
      "severity": 5,
      "suggested_fix": "Clarify: 'Each layer has weights, biases, and an activation function. By stacking multiple such layers, even with non-linearity between each layer, the network as a whole can learn patterns that a single linear layer cannot express.'",
      "consensus_models": ["claude", "gpt-4o-mini", "deepseek", "llama"]
    }
  ],
  "consensus_blockers": [
    {
      "claim": "Neurons with different activation functions can learn different things",
      "conflicting_positions": {
        "claude": [
          "Activation function choice affects training dynamics but all activations in a single layer typically match.",
          "Activation function choice affects training dynamics but all activations in a single layer typically match.",
          "Activation function choice affects training dynamics but all activations in a single layer typically match.",
          "Activation function choice affects training dynamics but all activations in a single layer typically match.",
          "Activation function choice affects training dynamics but all activations in a single layer typically match."
        ],
        "gpt-4o-mini": [
          "Different activations allow different representational capacity; you can mix them.",
          "Different activations allow different representational capacity; you can mix them.",
          "Standard networks use uniform activation per layer, but mixed activations are theoretically possible.",
          "Standard networks use uniform activation per layer, but mixed activations are theoretically possible.",
          "I concur with Claude's clarification."
        ]
      },
      "rounds_debated": 5
    }
  ],
  "cost_estimate": 8.75
}
```

Status = NEEDS_REVISION because 2 agreed items have severity > 2.

One blocker remains: the two models couldn't agree on whether the article's phrasing about mixing activation functions is valid pedagogy or imprecise.

## Next Steps

To fact-check an article:

1. Install dependencies: `pip install -r requirements_fact_check.txt`
2. Set environment variables for API keys
3. Run: `python fact_check.py <path_to_article>`
4. Review the output JSON
5. For NEEDS_REVISION articles, apply the suggested fixes and re-run
6. For blockers, manually review conflicting positions and decide which model's view is right for your audience
