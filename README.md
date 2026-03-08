# CoCo — Collaborate || Compete

<p align="center">
  <strong>An Evolutionary Agentic LLM Framework</strong>
</p>

<p align="center">
  <a href="https://github.com/pagel-s/coco/actions/workflows/ci.yml?branch=main"><img src="https://img.shields.io/github/actions/workflow/status/pagel-s/coco/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=for-the-badge"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=for-the-badge" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**CoCo** is an agentic LLM framework designed for agent-based optimization of arbitrary objectives. In this framework, a set of AI agents dynamically evolve and interact within an environment, choosing to either **collaborate** together towards a shared goal or **compete** (and even steal resources/information) for individual fitness.

Inspired by projects like OpenClaw, CoCo provides a generalized sandbox for emergent, Darwinian behavior among LLM-driven agents.

## Highlights

- **Dynamic Social Interaction** — Agents can form alliances, share contexts, trade resources, or betray each other.
- **Theft & Sabotage Mechanics** — Robust environment-level enforcement of stealing resources (compute tokens, hints, data fragments) to reward cunning strategies.
- **Evolutionary Loop** — Agents mutate over generations based on a fitness function. Successful traits (trust levels, collaboration thresholds, aggression) and prompts are passed down and modified.
- **Model-Agnostic** — Powered by `litellm`, allowing agents to be powered by OpenAI, Anthropic, local models, or a mixture of different models in the same environment.
- **Asynchronous Engine** — Fully `asyncio` driven, allowing for real-time, non-turn-based agent interactions where speed can be a competitive advantage.

## How it works

1. **The Environment (The Arena)**: Acts as the source of truth, managing the state, defining tasks, and serving as an un-cheatable ledger for resources.
2. **The Agents**: Each agent possesses:
    - **LLM Core**: For reasoning and decision-making.
    - **Traits**: Inherent thresholds (e.g., `trust_level`, `aggression_threshold`) that mutate across generations.
    - **Action Space**: Capable of interacting with the world (reading files, executing code) and other agents (sharing, stealing, messaging).
3. **The Evolutionary Engine**: Evaluates agents based on their success at tasks, culling underperformers and generating new offspring by crossing over and mutating the traits of top performers.

## Install (recommended)

Runtime: **Python ≥3.10**.

```bash
git clone https://github.com/pagel-s/coco.git
cd coco

# Install dependencies
pip install -e .
```

To install development dependencies (for running tests, linting, etc.):

```bash
pip install -e .[dev]
```

## Quick start

*Coming soon. We are actively developing the first scenario.*

The initial testbed will focus on a resource-constrained optimization task, where agents must decide whether to pool their limited tokens to solve a complex coding task together, or steal tokens from others to complete it solo.

## Development & CI/CD

CoCo enforces high-quality engineering standards:
- **Type Hinting**: Checked via `mypy`.
- **Linting/Formatting**: Checked via `ruff`.
- **Testing**: Comprehensive test suite run via `pytest`.

To run the full suite locally:

```bash
ruff check .
mypy coco/
pytest tests/
```

We utilize GitHub Actions for our CI/CD pipeline, automatically validating all pull requests and merges to `main`.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page.

## License

This project is licensed under the MIT License.
