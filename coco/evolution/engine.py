from typing import List, Dict, Any, Type, Optional
import random

from coco.core.agent import Agent, AgentTraits
from coco.core.environment import Environment

class EvolutionaryEngine:
    """
    Manages the lifecycle of multiple generations of agents.
    Evaluates fitness, culls underperformers, and breeds/mutates new agents.
    """
    def __init__(
        self, 
        environment_class: Type[Environment],
        population_size: int = 10,
        survival_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_step: float = 0.2,
        model: str = "gpt-3.5-turbo",
        data_manager: Optional[Any] = None,
        simulation_id: Optional[int] = None
    ):
        self.environment_class = environment_class
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.model = model
        self.data_manager = data_manager
        self.simulation_id = simulation_id
        
        self.generation = 0
        self.population: List[Agent] = self._initialize_population()
        self.history_log: List[Dict[str, Any]] = []

    def _initialize_population(self) -> List[Agent]:
        """Creates Gen 0 with completely random traits."""
        population = []
        for i in range(self.population_size):
            traits = AgentTraits(
                collaboration_threshold=random.random(),
                aggression_threshold=random.random(),
                trust_level=random.random()
            )
            agent = Agent(
                agent_id=f"gen_{self.generation}_agent_{i}", 
                traits=traits, 
                model=self.model,
                generation=self.generation
            )
            population.append(agent)
            
            # Log to DB
            if self.data_manager and self.simulation_id is not None:
                self.data_manager.log_agent(self.simulation_id, agent)
                
        return population

    def _mutate_trait(self, value: float) -> float:
        """Slightly shifts a trait value up or down, bounded between 0.0 and 1.0."""
        if random.random() < self.mutation_rate:
            shift = (random.random() * 2 * self.mutation_step) - self.mutation_step
            new_value = value + shift
            return max(0.0, min(1.0, new_value))
        return value

    def breed(self, parent_a: Agent, parent_b: Agent, child_id: str) -> Agent:
        """Creates a new agent by combining traits of two parents and applying mutation."""
        child_traits = AgentTraits(
            collaboration_threshold=parent_a.traits.collaboration_threshold if random.random() > 0.5 else parent_b.traits.collaboration_threshold,
            aggression_threshold=parent_a.traits.aggression_threshold if random.random() > 0.5 else parent_b.traits.aggression_threshold,
            trust_level=parent_a.traits.trust_level if random.random() > 0.5 else parent_b.traits.trust_level
        )
        
        child_traits.collaboration_threshold = self._mutate_trait(child_traits.collaboration_threshold)
        child_traits.aggression_threshold = self._mutate_trait(child_traits.aggression_threshold)
        child_traits.trust_level = self._mutate_trait(child_traits.trust_level)
        
        # Primary parent (parent_a) for lineage tracking
        return Agent(
            agent_id=child_id, 
            traits=child_traits, 
            model=self.model,
            generation=self.generation + 1,
            parent_id=parent_a.agent_id
        )

    def evolve(self) -> None:
        """
        Sorts the current population by fitness, kills the bottom percentage, 
        and breeds the survivors to fill the population back up.
        """
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        
        num_survivors = int(self.population_size * self.survival_rate)
        survivors = self.population[:num_survivors]
        
        avg_fitness = sum(a.fitness for a in self.population) / self.population_size
        avg_aggression = sum(a.traits.aggression_threshold for a in self.population) / self.population_size
        avg_collaboration = sum(a.traits.collaboration_threshold for a in self.population) / self.population_size
        
        self.history_log.append({
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "avg_aggression": avg_aggression,
            "avg_collaboration": avg_collaboration
        })
        
        new_population = list(survivors)
        
        while len(new_population) < self.population_size:
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)
            child_id = f"gen_{self.generation+1}_agent_{len(new_population)}"
            child = self.breed(parent_a, parent_b, child_id)
            new_population.append(child)
            
            # Log new offspring to DB
            if self.data_manager and self.simulation_id is not None:
                self.data_manager.log_agent(self.simulation_id, child, parent_id=parent_a.agent_id)
            
        for agent in new_population:
            agent.fitness = 0.0
            agent.resources = {}
            agent.memory = []
            
        self.population = new_population
        self.generation += 1
