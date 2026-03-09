# CoCo — Collaborate || Compete

<p align="center">
  <strong>An Evolutionary Agentic LLM Framework</strong>
</p>

<p align="center">
  <a href="https://github.com/pagel-s/coco/actions/workflows/ci.yml?branch=main"><img src="https://img.shields.io/github/actions/workflow/status/pagel-s/coco/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg?style=for-the-badge"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg?style=for-the-badge" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**CoCo** is a professional-grade agentic LLM framework designed for the evolutionary optimization of arbitrary objectives. It provides a sandbox where agents dynamically choose between collaboration and competition, evolving their behavioral traits over generations.

## Highlights

- **Dynamic Social Interaction** — Agents form alliances, share contexts, trade resources, or betray each other.
- **Theft & Sabotage Mechanics** — Environment-level enforcement of stealing resources (tokens, snippets) to reward cunning strategies.
- **Evolutionary Loop** — Agents mutate over generations based on task-specific fitness functions.
- **Associative Memory** — Two-tier memory system including Working Memory and Long-term Associative Memory powered by VectorDB (ChromaDB).
- **Model-Agnostic** — Powered by `litellm`, supporting OpenAI, Anthropic, Ollama, and more.
- **Asynchronous Engine** — Fully `asyncio` driven for real-time, non-turn-based agent interactions.

## Core Architecture

1. **The Environment**: The un-cheatable "physics engine" managing the state and resource ledger.
2. **The Agents**: Autonomous reasoning nodes with mutable genomes (collaboration, aggression, trust).
3. **The Evolutionary Engine**: Manages the Darwinian loop (Selection, Crossover, Mutation).
4. **Analysis Layer**: Automatic SQLite logging of every turn, interaction, and reasoning trace.

## Install

Runtime: **Python ≥3.11**. Recommended management via `uv`.

```bash
git clone https://github.com/pagel-s/coco.git
cd coco

# Install package and dependencies
uv pip install -e .
```

To install development dependencies (testing, linting):

```bash
uv pip install -e .[dev]
```

## Quick Start (CLI)

CoCo provides easy-to-use CLI entry points:

```bash
# Run the default evolutionary Token Heist simulation
coco sim

# Run the Collaborative Code Fix task
coco codefix

# Launch the interactive analysis dashboard
coco dashboard
```

## Benchmarks

- **The Token Heist**: A zero-sum survival scenario where agents must manage limited tokens.
- **Collaborative Code Fix**: Agents must fix buggy Python methods by sharing or stealing code snippets.

## Analysis Dashboard

View results in a high-quality Streamlit UI:
- Visualized Social Network Graphs (who stole from whom).
- Trait Evolution charts across generations.
- Deep-dive into raw LLM reasoning and private memories.

## Engineering Standards

- **Type Hinting**: 100% strict `mypy` compliance.
- **Testing**: ~90% unit test coverage with `pytest`.
- **CI/CD**: Fully automated via GitHub Actions.
- **Linter**: Blazing fast enforcement with `ruff`.

## License

This project is licensed under the MIT License.
