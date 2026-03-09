# CoCo: Collaborate || Compete — Comprehensive Documentation

CoCo is a professional-grade agentic LLM framework designed for the evolutionary optimization of arbitrary objectives. It provides a sandbox where agents dynamically choose between collaboration and competition, evolving their behavioral traits over generations.

---

## 🏗️ Core Architecture

The system is built on a strict separation of concerns between physical world rules (Environment) and cognitive logic (Agents).

### 1. The Environment (`coco.core.environment`)
The Environment is the un-cheatable "physics engine" of the simulation.
- **Resource Ledger:** A transactional source of truth tracking agent assets (tokens, code snippets, memory fragments).
- **Conflict Resolution:** Handles logic for `steal` and `share` actions, including probability-based theft success.
- **Orchestration:** Manages asynchronous execution of agent turns using `asyncio.gather`.
- **Hooks:** Provides `_pre_step` and `_post_step` hooks for task-specific logic like token consumption or win-condition checking.

### 2. The Agent (`coco.core.agent`)
Each agent is an autonomous reasoning node powered by an LLM.
- **Genome (Traits):** Mutable thresholds [0.0, 1.0] for `collaboration_threshold`, `aggression_threshold`, and `trust_level`.
- **Cognitive Core:** Uses `litellm` to process environment state and output structured JSON actions.
- **Memory Systems:** 
    - **Working Memory:** The last 5 turns provided literally in the prompt.
    - **Associative Memory (VectorDB):** Uses `ChromaDB` to perform semantic retrieval of relevant past experiences.
- **Social Ledger:** Tracks local trust scores for other agents in the environment.

### 3. The Evolutionary Engine (`coco.evolution.engine`)
Manages the Darwinian loop across multiple generations.
- **Selection:** Ranks agents based on a task-specific `fitness` score.
- **Culling:** Removes the bottom performers based on a configurable `survival_rate`.
- **Breeding:** Offspring inherit traits via crossover from top performers.
- **Mutation:** Small random shifts applied to traits to explore new behavioral strategies.

### 4. Data & Analysis Layer (`coco.core.database`)
- **SQLite DataManager:** Automatically logs every simulation metadata, turn state, interaction (source, target, reasoning), and agent snapshot.
- **Lineage Tracking:** Records parent-child relationships for genealogical analysis.

---

## 🎮 Implemented Tasks (Benchmarks)

### 🏦 The Token Heist (`coco.tasks.token_heist`)
A zero-sum survival benchmark.
- **Objective:** Survive as long as possible with limited tokens.
- **Dynamics:** 1 token consumed per turn. Agents must decide whether to pool tokens to keep everyone alive or steal tokens from others to ensure personal survival.

### 💻 Collaborative Code Fix (`coco.tasks.code_fix`)
A complex text-based optimization task.
- **Objective:** Fix buggy Python methods.
- **Dynamics:** Submit fixes for rewards. Agents can choose to `share_snippet` publicly for a small bonus or `steal_snippet` from others to claim the high-value solution reward for themselves.

### 🎲 Number Guesser (`coco.tasks.number_guesser`)
A minimal "Hello World" task used to verify the LLM integration and prompt adherence.

---

## 📊 Analysis Tooling

CoCo includes a professional **Streamlit Dashboard** (`coco/analysis/app.py`) for deep-dive research:
- **Evolutionary View:** Charts showing the rise/fall of Aggression vs. Collaboration over 100 generations.
- **Social Network Graph:** A directed graph of all `steal` and `share` interactions per generation.
- **Reasoning Trace:** Read the raw internal logic written by the LLM during every turn.
- **Agent Deep-Dive:** Track a specific lineage's resource history and private memory.

---

## 🛠️ Engineering Standards

- **Environment:** Managed by `uv` for high performance.
- **Type Safety:** 100% `mypy` strict mode compliance.
- **Testing:** ~90% coverage using `pytest` and `pytest-asyncio`.
- **CI/CD:** Automated GitHub Actions pipeline for every push.
- **Linter:** `ruff` for extremely fast code quality enforcement.

---

## 🚀 Quick Start Commands

**Run a Simulation:**
```bash
uv run python3 examples/run_evolution_token_heist.py
```

**Launch Analysis Dashboard:**
```bash
uv run streamlit run coco/analysis/app.py
```

**Run Tests & Coverage:**
```bash
uv run pytest
```
