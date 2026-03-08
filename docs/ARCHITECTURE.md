# CoCo Architecture

## Core Philosophy
CoCo separates the "physical" rules of the world from the cognitive logic of the agents. Agents interact with the world via strict interfaces; the Environment is the absolute source of truth.

## System Components

### 1. The Environment (`coco.core.environment`)
The arena where tasks are executed. It ensures state integrity and prevents agent hallucination/cheating.
- **State Manager:** Holds the absolute state of the puzzle/task.
- **Resource Ledger:** A strict transactional record of which agent owns what (e.g., compute tokens, hints).
- **Physics Engine:** Resolves conflicts (e.g., if Agent A tries to steal from Agent B, the environment rolls the dice and executes the transfer).

### 2. The Agent (`coco.core.agent`)
The individual participant. It perceives the environment, reasons via an LLM, and takes actions.
- **Cognitive Core (LLM):** Uses `litellm` to process state and generate action JSONs.
- **Traits (`AgentTraits`):** Continuous variables [0.0, 1.0] defining behavior (collaboration, aggression, trust, greed).
- **Memory:** 
  - *Short-term:* Current epoch scratchpad.
  - *Long-term:* Summaries of past epochs/interactions with specific other agents.
- **Action Space:**
  - `Environment Actions`: Execute code, submit answer, read task data.
  - `Social Actions`: Share resource, steal resource, send message.

### 3. The Evolutionary Engine (`coco.evolution`)
The Darwinian loop that operates outside the environment.
- **Fitness Evaluation:** At the end of an epoch, agents are scored based on the specific Task's objective.
- **Selection:** Culling the lowest performers (e.g., bottom 50%).
- **Crossover/Mutation:** Generating new agents by combining the Traits and prompt-structures of the top performers.

### 4. Tasks (`coco.tasks`)
Pluggable objectives. A task defines:
- What resources exist.
- What constitutes "success" (the fitness function).
- Example: "The Token Heist" - Agents start with 10 tokens. Executing a line of code costs 1 token. They must solve a math problem. They can steal tokens to survive or pool tokens to solve it faster.
