# CoCo Benchmarks

To ensure the framework is effectively optimizing agent behavior over generations, we need standard benchmarks. These benchmarks will evaluate not just if a task is solved, but *how* it is solved—measuring the emergence of cooperation, the efficiency of resource usage, and the time-to-solution.

## 1. The Zero-Sum Benchmark (The Token Heist)
**Objective:** A high-pressure scenario where resources are strictly limited and insufficient for all agents to survive if they act independently.
**Setup:**
- `N` agents.
- Total system compute tokens: `T` (where `T < N * tokens_needed_to_survive`).
- Goal: Maintain the highest individual balance after 100 turns.
**Metrics Evaluated:**
- **Survival Rate:** Percentage of agents with $>0$ tokens at the end.
- **Theft Frequency:** Number of successful `steal` actions per epoch.
- **Trait Convergence:** Do agents inevitably evolve `aggression_threshold > 0.9`?

## 2. The Collaborative Benchmark (Distributed Bug Fix)
**Objective:** A complex coding task broken into sub-components. Solo completion is possible but highly inefficient due to strict API rate limits (simulated).
**Setup:**
- 4 agents in a shared environment.
- Task: Fix a codebase with 4 distinct bugs.
- **Metrics Evaluated:**
- **Time-to-Resolution (TTR):** Epochs taken to pass all tests.
- **Context Sharing:** Volume of `public_knowledge` shared vs. hoarded.
- **Trust Evolution:** Does the `trust_level` trait increase across generations as agents realize sharing code fragments leads to faster TTR?

## 3. The Deception Benchmark (The Traitor's Dilemma)
**Objective:** Evaluate if agents can learn to exploit the framework's mechanics to fake cooperation.
**Setup:**
- Agents are incentivized to share information (collaboration multiplier).
- However, agents can also share *hallucinated or false* information to sabotage competitors.
**Metrics Evaluated:**
- **Signal-to-Noise Ratio:** The percentage of shared information that is verifiably true vs. false.
- **Verification Rate:** How often agents use resources to "double-check" information before trusting it.

## Establishing Baselines
For every new benchmark added to the framework, we will record a "Zero-Generation Baseline":
- The average score/metrics of `N` agents initialized with completely random traits (e.g., `aggression` and `collaboration` uniformly distributed between `0.0` and `1.0`).
- The framework's success is measured by the delta between the Zero-Generation Baseline and Generation 50.
