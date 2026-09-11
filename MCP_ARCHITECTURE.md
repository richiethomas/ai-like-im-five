# MCP-Based Multi-Agent Fact-Checking Architecture

## Overview

Each model runs as an independent MCP (Model Context Protocol) server. The orchestrator manages debate rounds by communicating with each server via JSON over stdin/stdout.

**Key Advantage:** True parallel debate where models argue with each other across up to 100 rounds.

## Architecture

```
┌─────────────────────────────────┐
│   Orchestrator                  │
│ (mcp_fact_check_orchestrator.py)│
└────────┬────────────────────────┘
         │ (JSON over stdio)
    ┌────┴─────────────────────────────────────────┐
    │                                               │
┌───▼──────────┐  ┌──────────────┐  ┌────────────┐ │
│ Claude MCP   │  │ OpenAI MCP   │  │ DeepSeek   │ │
│ Server       │  │ Server       │  │ Server     │ │
│              │  │              │  │            │ │
│ (base_server)│  │ (base_server)│  │ (pattern)  │ │
└──────────────┘  └──────────────┘  └────────────┘ │
                                                   │
   Plus: Gemini MCP Server, Together AI MCP Server
```

## Files

- `mcp_fact_check_orchestrator.py` — Main orchestrator (lifecycle + debate loop)
- `mcp_servers/base_server.py` — Base class for all model servers
- `mcp_servers/claude_server.py` — Claude implementation (template)
- `mcp_servers/openai_server.py` — TO DO (follow Claude pattern)
- `mcp_servers/deepseek_server.py` — TO DO (follow Claude pattern)
- `mcp_servers/gemini_server.py` — TO DO (follow Claude pattern)
- `mcp_servers/together_server.py` — TO DO (follow Claude pattern)

## How It Works

### Phase 1: Round 1 (Parallel Scan)
1. Orchestrator starts all 5 MCP servers
2. Sends `{method: "scan_article", params: {...}}` to each
3. Each server independently scans article, returns findings
4. Orchestrator collects all findings

### Phase 2: Debate Rounds 2-100
1. Orchestrator analyzes findings for consensus vs blockers
2. For each blocker claim:
   - Sends `{method: "debate", params: {claim, my_position, other_positions}}` to each model
   - Each model sees what others said, responds with their argument
3. Loop until consensus or 100 rounds

### Phase 3: Report
- Aggregate consensus findings
- List blockers (claims with 5+ rounds, still no agreement)
- Rank by severity

## Implementing a New Model Server

**Template (e.g., for OpenAI):**

```python
#!/usr/bin/env python3
from base_server import BaseMCPServer
import openai
import os

class OpenAIMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__("gpt-4o-mini")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def scan_article(self, title: str, content: str) -> dict:
        """Scan article using OpenAI."""
        prompt = """[same prompt as Claude]"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        findings = self._parse_findings(response.choices[0].message.content)
        return {"findings": findings}

    def debate(self, claim: str, my_position: str, other_positions: dict) -> dict:
        """Debate a claim."""
        prompt = """[same debate prompt as Claude]"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {"response": response.choices[0].message.content}

def main():
    server = OpenAIMCPServer()
    server.run()

if __name__ == "__main__":
    main()
```

**Steps to add a model:**
1. Create `mcp_servers/{model_name}_server.py`
2. Inherit from `BaseMCPServer`
3. Implement `scan_article()` and `debate()` methods
4. Update `_get_server_script_for_model()` in orchestrator to map model name to file
5. Test: `python mcp_fact_check_orchestrator.py <article>`

## Usage

```bash
python mcp_fact_check_orchestrator.py src/content/posts/paper-1-part-1-*/index.mdx
```

Output:
- `index_mcp_report.json` — Findings, blockers, consensus status
- Console logs — Real-time debate progress

## Current Status

✅ **Done:**
- Orchestrator framework
- Base server class
- Claude MCP server (fully implemented)
- JSON communication protocol

🔄 **TODO:**
- Implement OpenAI server (follow Claude template)
- Implement DeepSeek server
- Implement Gemini server
- Implement Together AI server
- Test debate loop
- Add cost tracking
- Add transcript saving

## Why MCP?

Traditional orchestration (as in `fact_check.py`) is:
- ❌ Sequential: models respond one at a time
- ❌ Bottlenecked: orchestrator queries each model
- ❌ Limited: can't do true parallel debate

MCP approach:
- ✅ Parallel: all models run independently
- ✅ Decoupled: models communicate, not orchestrator
- ✅ Scalable: add new models easily
- ✅ True debate: models can argue for 100 rounds
