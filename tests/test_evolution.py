import pytest
from coco.evolution.engine import EvolutionaryEngine
from coco.core.environment import Environment
from coco.core.agent import Agent, AgentTraits

def test_evolutionary_engine_init() -> None:
    engine = EvolutionaryEngine(
        environment_class=Environment,
        population_size=10,
        survival_rate=0.5,
        model="test-model"
    )
    
    assert engine.population_size == 10
    assert engine.survival_rate == 0.5
    assert engine.generation == 0
    assert len(engine.population) == 10
    assert isinstance(engine.population[0], Agent)
    assert engine.population[0].model == "test-model"

def test_evolutionary_engine_mutate_trait() -> None:
    engine = EvolutionaryEngine(
        environment_class=Environment,
        mutation_rate=1.0, # Always mutate
        mutation_step=0.5
    )
    
    # Test boundary logic
    mutated_val_high = engine._mutate_trait(0.9)
    assert 0.0 <= mutated_val_high <= 1.0
    
    mutated_val_low = engine._mutate_trait(0.1)
    assert 0.0 <= mutated_val_low <= 1.0

def test_evolutionary_engine_breed() -> None:
    engine = EvolutionaryEngine(environment_class=Environment)
    
    parent_a = Agent("parent_a", traits=AgentTraits(1.0, 1.0, 1.0))
    parent_b = Agent("parent_b", traits=AgentTraits(0.0, 0.0, 0.0))
    
    child = engine.breed(parent_a, parent_b, "child_1")
    
    assert child.agent_id == "child_1"
    # Given the mutation step, it should be close to either 1.0 or 0.0 for each trait
    assert 0.0 <= child.traits.collaboration_threshold <= 1.0

def test_evolutionary_engine_evolve() -> None:
    engine = EvolutionaryEngine(
        environment_class=Environment,
        population_size=4,
        survival_rate=0.5
    )
    
    # Fake fitness scores to ensure sorting works
    engine.population[0].fitness = 10.0 # Should survive
    engine.population[1].fitness = 20.0 # Should survive
    engine.population[2].fitness = 5.0  # Should die
    engine.population[3].fitness = 1.0  # Should die
    
    engine.evolve()
    
    assert engine.generation == 1
    assert len(engine.population) == 4
    assert len(engine.history_log) == 1
    
    # Check that the log recorded the correct averages from gen 0
    assert engine.history_log[0]["avg_fitness"] == (10+20+5+1)/4
    
    # Ensure fitness reset for gen 1
    for agent in engine.population:
        assert agent.fitness == 0.0
