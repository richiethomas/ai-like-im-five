# Fact-Check Quickstart

## Setup (one time)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_fact_check.txt
   ```

2. **Set API keys** in your shell environment:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export DEEPSEEK_API_KEY="sk-..."      # optional (for DeepSeek)
   export GEMINI_API_KEY="..."           # optional (for Gemini)
   export LLAMA_API_KEY="..."            # optional (for Llama)
   ```

   Or add to `~/.zshrc` / `~/.bashrc` to persist them:
   ```bash
   echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.zshrc
   source ~/.zshrc
   ```

## Run a Fact-Check

**Via the shell script (recommended):**
```bash
./fact_check.sh src/content/posts/paper-1-part-8-neurons/index.mdx
```

**Or directly:**
```bash
python3 fact_check.py src/content/posts/paper-1-part-8-neurons/index.mdx
```

## Output

The script generates a JSON report next to the article:
```
src/content/posts/paper-1-part-8-neurons/index_fact_check.json
```

**Report structure:**
- `status`: "PASS" or "NEEDS_REVISION"
- `agreed_upon_changes`: list of issues to fix, sorted by severity
- `consensus_blockers`: items where models couldn't agree
- `cost_estimate`: total API spend in USD

## Interpreting Results

### PASS
No action items with severity > 2. The article is factually sound per the LLM consensus.

### NEEDS_REVISION
One or more issues with severity > 2. Review `agreed_upon_changes` and apply fixes:

```json
{
  "claim": "The ReLU function sets negative values to zero",
  "issue": "Imprecise phrasing; ReLU outputs zero but doesn't 'set' them",
  "severity": 4,
  "suggested_fix": "Change to: 'ReLU outputs zero for negative inputs...'",
  "consensus_models": ["claude", "gpt-4o-mini", "gemini"]
}
```

Edit the article, re-run the fact-check, and repeat until status is PASS.

### Consensus Blockers

Issues where LLMs debated 5 times with no resolution:

```json
{
  "claim": "Neurons can have different activation functions",
  "conflicting_positions": {
    "claude": ["Position A", "Position A", "Position A", "Position A", "Position A"],
    "gpt-4o-mini": ["Position B", "Position B", "Position B", "Position B", "Position B"]
  },
  "rounds_debated": 5
}
```

**What to do:** Manually decide which model's interpretation is right *for your audience* (non-technical readers). You may need to clarify the claim or simplify it.

## Tuning for Your Blog

### Cost Budget
Edit `fact_check.py`, line 54:
```python
self.budget_per_article = 10.0  # change to your budget in dollars
```

### Max Rounds
Edit `fact_check.py`, line 59:
```python
self.max_rounds = 100  # max debate iterations
```

### Severity Threshold
Edit `fact_check.py`, line 315:
```python
action_items = [item for item in consensus if item.severity > 2]  # change 2 to your threshold
```

Lower threshold (e.g., > 1) = more strict. Higher threshold (e.g., > 5) = more lenient.

## Workflow

**Per article:**
1. Write the article with `draft: true` in frontmatter
2. Run: `./fact_check.sh src/content/posts/article/index.mdx`
3. Wait ~15-30 minutes (depends on article length and model response times)
4. Check the generated JSON report
5. If NEEDS_REVISION, apply fixes and re-run
6. Once PASS, set `draft: false` and commit

**Batch checking:**
```bash
# Check all articles
for file in src/content/posts/*/index.mdx; do
  ./fact_check.sh "$file"
done
```

## Troubleshooting

**"Error: File not found"**
- Double-check the path: `ls -la src/content/posts/article/index.mdx`
- Run from project root

**"Missing API keys"**
- Set environment variables (see Setup section)
- Verify: `echo $ANTHROPIC_API_KEY` (should print your key)

**"Module not found: anthropic"**
- Install dependencies: `pip install -r requirements_fact_check.txt`

**Script hangs / takes very long**
- Normal for first round (scanning 5 models in parallel)
- Debate rounds are sequential (slower)
- Very long articles (~5000 chars) take longer
- Check if a model API is slow/down

**Cost estimate too high**
- Reduce `max_rounds` in `fact_check.py`
- Or reduce article length in `_initial_fact_check` (line 131: `content[:3000]` → `content[:2000]`)

## Example Session

```bash
$ ./fact_check.sh src/content/posts/paper-1-part-8-neurons/index.mdx

🔍 Fact-checking: src/content/posts/paper-1-part-8-neurons/index.mdx
============================================

============================================================
Fact-checking: Paper #1, Part 8: Neurons and Networks
============================================================

Round 1: Initial fact-checks...
  claude-3-5-sonnet-20241022: 4 findings
  gpt-4o-mini: 3 findings
  deepseek-chat: 4 findings
  gemini-2.0-flash: 2 findings

Round 2: Debate round...
  Active disagreements: 2

Round 3: Debate round...
  Active disagreements: 2

Round 4: Debate round...
  Active disagreements: 0

✓ Consensus reached in round 4

============================================================
Report saved to: src/content/posts/paper-1-part-8-neurons/index_fact_check.json
Status: NEEDS_REVISION
Action items (severity > 2): 2
Consensus blockers: 0
Estimated cost: $3.45
============================================================

📄 Full report: src/content/posts/paper-1-part-8-neurons/index_fact_check.json

Status: NEEDS_REVISION
Action items (severity > 2): 2
Consensus blockers: 0
```

Then review the JSON, fix the 2 action items, and re-run.

---

**Questions?** Check `FACT_CHECK_ARCHITECTURE.md` for detailed design docs.
