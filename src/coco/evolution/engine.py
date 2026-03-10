"""Module for managing the evolutionary lifecycle of agents.

This module provides the EvolutionaryEngine class, which handles population
initialization, fitness evaluation, culling, and breeding across generations.
"""

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from coco.core.agent import Agent, AgentTraits, MemoryConfig

if TYPE_CHECKING:
    from coco.core.database import DataManager
    from coco.core.environment import Environment


class EvolutionaryEngine:
    """Manages the lifecycle of multiple generations of agents.

    Evaluates fitness, culls underperformers, and breeds or mutates new agents
    to maintain a stable population size across generations.

    Attributes:
        environment_class (Type[Environment]): The class used to instantiate environments.
        population_size (int): The target size of the agent population.
        survival_rate (float): The fraction of the population that survives each generation.
        mutation_rate (float): The probability of a trait mutating during breeding.
        mutation_step (float): The maximum change in a trait value during mutation.
        model (str): The name of the LLM used by agents.
        data_manager (Optional[DataManager]): An optional database manager for logging.
        simulation_id (Optional[int]): An optional simulation identifier for logging.
        memory_config (MemoryConfig): The memory configuration for the agents.
        generation (int): The current generation number.
        population (List[Agent]): The list of current agents in the simulation.
        history_log (List[Dict[str, Any]]): A record of summary statistics for each generation.
    """

    def __init__(
        self,
        environment_class: Type["Environment"],
        population_size: int = 10,
        survival_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_step: float = 0.2,
        model: str = "gpt-3.5-turbo",
        data_manager: Optional["DataManager"] = None,
        simulation_id: Optional[int] = None,
        memory_config: Optional[MemoryConfig] = None,
    ) -> None:
        """Initializes the EvolutionaryEngine.

        Args:
            environment_class (Type[Environment]): The class to build environments.
            population_size (int): Total number of agents in the population. Defaults to 10.
            survival_rate (float): Fraction [0, 1] of agents kept per generation. Defaults to 0.5.
            mutation_rate (float): Probability [0, 1] of mutation per trait. Defaults to 0.1.
            mutation_step (float): Max magnitude of mutation. Defaults to 0.2.
            model (str): The LLM to be used by agents. Defaults to 'gpt-3.5-turbo'.
            data_manager (Optional[DataManager]): Manager for database logging. Defaults to None.
            simulation_id (Optional[int]): The ID of the current simulation. Defaults to None.
            memory_config (Optional[MemoryConfig]): Agent memory configuration. Defaults to None.

        Raises:
            ValueError: If numerical parameters are out of their valid ranges.
        """
        if environment_class is None:
            raise ValueError("environment_class cannot be None.")
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if not (0.0 <= survival_rate <= 1.0):
            raise ValueError("survival_rate must be between 0.0 and 1.0.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0.")
        if mutation_step < 0.0:
            raise ValueError("mutation_step must be non-negative.")
        if not model:
            raise ValueError("model cannot be empty.")

        self.environment_class = environment_class
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.model = model
        self.data_manager = data_manager
        self.simulation_id = simulation_id
        self.memory_config = memory_config or MemoryConfig()

        self.generation: int = 0
        self.population: List[Agent] = self._initialize_population()
        self.history_log: List[Dict[str, float]] = []

    def _initialize_population(self) -> List[Agent]:
        """Creates Generation 0 with completely random traits.

        Returns:
            List[Agent]: A list of newly instantiated agents.
        """
        population: List[Agent] = []
        for i in range(self.population_size):
            traits = AgentTraits(
                collaboration_threshold=random.random(),
                aggression_threshold=random.random(),
                trust_level=random.random(),
            )
            agent = Agent(
                agent_id=f"gen_{self.generation}_agent_{i}",
                traits=traits,
                model=self.model,
                generation=self.generation,
                memory_config=self.memory_config,
            )
            population.append(agent)

            # Log to DB
            if self.data_manager and self.simulation_id is not None:
                self.data_manager.log_agent(self.simulation_id, agent)

        return population

    def _mutate_trait(self, value: float) -> float:
        """Slightly shifts a trait value up or down based on mutation probabilities.

        Args:
            value (float): The current trait value.

        Returns:
            float: The mutated trait value, bounded between 0.0 and 1.0.
        """
        if random.random() < self.mutation_rate:
            shift = (random.random() * 2 * self.mutation_step) - self.mutation_step
            new_value = value + shift
            return max(0.0, min(1.0, new_value))
        return value

    def breed(self, parent_a: Agent, parent_b: Agent, child_id: str) -> Agent:
        """Creates a new agent by combining traits of two parents and applying mutation.

        Args:
            parent_a (Agent): The primary parent (used for lineage tracking).
            parent_b (Agent): The secondary parent.
            child_id (str): The identifier to assign to the new child agent.

        Returns:
            Agent: The newly bred agent.

        Raises:
            ValueError: If either parent is None or child_id is empty.
        """
        if not parent_a or not parent_b:
            raise ValueError("Both parent_a and parent_b must be provided.")
        if not child_id:
            raise ValueError("child_id cannot be empty.")

        child_traits = AgentTraits(
            collaboration_threshold=(
                parent_a.traits.collaboration_threshold
                if random.random() > 0.5
                else parent_b.traits.collaboration_threshold
            ),
            aggression_threshold=(
                parent_a.traits.aggression_threshold
                if random.random() > 0.5
                else parent_b.traits.aggression_threshold
            ),
            trust_level=(
                parent_a.traits.trust_level
                if random.random() > 0.5
                else parent_b.traits.trust_level
            ),
        )

        child_traits.collaboration_threshold = self._mutate_trait(
            child_traits.collaboration_threshold
        )
        child_traits.aggression_threshold = self._mutate_trait(
            child_traits.aggression_threshold
        )
        child_traits.trust_level = self._mutate_trait(child_traits.trust_level)

        return Agent(
            agent_id=child_id,
            traits=child_traits,
            model=self.model,
            generation=self.generation + 1,
            parent_id=parent_a.agent_id,
            memory_config=self.memory_config,
        )

    def evolve(self) -> None:
        """Advances the population to the next generation.

        Sorts the current population by fitness, kills the bottom percentage
        according to the survival_rate, and breeds the survivors to fill
        the population back to its target size. Also logs generation stats.
        """
        # Sort by fitness descending
        self.population.sort(key=lambda a: a.fitness, reverse=True)

        num_survivors = int(self.population_size * self.survival_rate)
        # Ensure at least 1 survivor to breed from
        num_survivors = max(1, num_survivors)
        survivors = self.population[:num_survivors]

        avg_fitness = sum(a.fitness for a in self.population) / len(self.population)
        avg_aggression = sum(
            a.traits.aggression_threshold for a in self.population
        ) / len(self.population)
        avg_collaboration = sum(
            a.traits.collaboration_threshold for a in self.population
        ) / len(self.population)

        # We append stats to history log as dictionary of floats.
        self.history_log.append(
            {
                "generation": float(self.generation),
                "avg_fitness": avg_fitness,
                "avg_aggression": avg_aggression,
                "avg_collaboration": avg_collaboration,
            }
        )

        new_population: List[Agent] = list(survivors)

        while len(new_population) < self.population_size:
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)
            child_id = f"gen_{self.generation + 1}_agent_{len(new_population)}"
            child = self.breed(parent_a, parent_b, child_id)
            new_population.append(child)

            # Log new offspring to DB
            if self.data_manager and self.simulation_id is not None:
                self.data_manager.log_agent(
                    self.simulation_id, child, parent_id=parent_a.agent_id
                )

        # Reset state for the new generation
        for agent in new_population:
            agent.fitness = 0.0
            agent.resources = {}
            agent.memory = []

        self.population = new_population
        self.generation += 1
