# CoCo Development Roadmap

## Phase 1: Foundation (Current)
**Goal:** Establish a single-player baseline where an agent can reason, interact with the environment, and complete a trivial task.

- [x] Scaffold repository, CI/CD, and professional documentation.
- [x] Define base classes (`Agent`, `Environment`, `AgentTraits`).
- [ ] **LLM Integration:** Implement `litellm` in `Agent.act()`. The agent must ingest the environment state as a prompt and output a structured JSON action.
- [ ] **First Toy Task:** Create `tasks/number_guesser.py`. A simple game where the agent must guess a number, but each guess costs a "compute token" resource.
- [ ] **Action Parser:** Build the mechanism in the `Environment` to parse the LLM's JSON output and execute the corresponding function (e.g., `guess_number(5)`).

## Phase 2: The Social Arena
**Goal:** Introduce multiple agents and the mechanics of Collaboration and Competition.

- [ ] **Messaging Protocol:** Implement an action allowing agents to broadcast messages or whisper to specific agents.
- [ ] **Resource Transfer (Collaboration):** Finalize the `share` action. Agents can voluntarily give resources (like compute tokens) to others.
- [ ] **Theft & Defense (Competition):** Finalize the `steal` action. Implement a basic probability model where theft success depends on the thief's `aggression` vs. the victim's `trust` or "defense" traits.
- [ ] **Task 2 (The Prisoners Dilemma / Token Heist):** A scenario where two or more agents have limited resources. They can only solve the task by pooling resources, *or* one agent can steal all resources from the other to win solo.

## Phase 3: Darwinian Evolution
**Goal:** Introduce epochs, scoring, and genetic mutation.

- [ ] **Fitness Functions:** Ensure tasks return a strict float score for each agent at the end of a run.
- [ ] **The Evolutionary Loop:** Implement `evolution/engine.py`.
  - Run an epoch (e.g., 10 turns of the Environment).
  - Score agents.
  - Cull the bottom 50%.
- [ ] **Mutation & Crossover:** Spawn new agents.
  - Mutate traits (e.g., `aggression` shifts from 0.8 to 0.85).
  - Mutate prompts (optional stretch goal: use an LLM to slightly rewrite the system prompt of the winning agents).

## Phase 4: Scaling & Complex Tasks
**Goal:** Make the framework useful for actual software engineering optimization.

- [ ] **Real-world Task:** "Collaborative Bug Fixing". Agents are given a broken Python file. They must write tests, share partial solutions, or steal working snippets from other agents' public scratchpads to fix the bug fastest.
- [ ] **Metrics & Visualization:** Implement logging to track how traits evolve over time (e.g., "Do agents become more aggressive or more collaborative in Task X?").
