"""Implements the simple example from the "Optimization and Simulation" course"""

import logging
from enum import Enum
from random import choice

import numpy as np
from teaching_optimization.metropolis_hastings.markov_chain import State
from teaching_optimization.metropolis_hastings.metropolis_hastings import (
    metropolis_hastings_algorithm,
)

logging.basicConfig(level=logging.INFO, format='%(message)s ')
logger = logging.getLogger('markov_chain')
logger.info('Test MH algorithm with the simple example from the lecture.')


class MachineCondition(Enum):
    """States of the machine (example from the lecture)"""

    STATE_A = 'Perfect condition'
    STATE_B = 'Partially damaged'
    STATE_C = 'Seriously damaged'
    STATE_D = 'Completely useless'


target_probabilities = {
    'Perfect condition': 5 / 8,
    'Partially damaged': 1 / 4,
    'Seriously damaged': 3 / 32,
    'Completely useless': 1 / 32,
}


def get_random_machine_condition() -> MachineCondition:
    """Select one randon state"""
    return choice(list(MachineCondition))


class SimpleMarkovState(State):
    """Simple state for a Markov Chain, that is a random walk among 4 states."""

    def __init__(self, the_state: MachineCondition) -> None:

        self.state: MachineCondition = the_state

    def indicators(self) -> dict[str, float]:
        """Relevant indicators associated with a state."""

        indicators = {condition.value: 0.0 for condition in MachineCondition}
        indicators[self.state.value] = 1.0
        return indicators


def proposal_state(
    current_state: SimpleMarkovState,
) -> tuple[SimpleMarkovState, float, float]:
    """
    Returns the next state of the Markov Chain

    :return: tuple with three elements: the next state, the log of the forward transition probability,
    the log of the backward transition probability
    """

    return SimpleMarkovState(the_state=get_random_machine_condition()), 0.25, 0.25


def log_target_probability(the_state: SimpleMarkovState) -> float:

    return np.log(target_probabilities[the_state.state.value])


initial_states = [
    SimpleMarkovState(the_state=MachineCondition.STATE_A),
    SimpleMarkovState(the_state=MachineCondition.STATE_B),
    SimpleMarkovState(the_state=MachineCondition.STATE_C),
    SimpleMarkovState(the_state=MachineCondition.STATE_D),
]

simulated_values = metropolis_hastings_algorithm(
    proposal_chain=proposal_state,
    log_target_probability=log_target_probability,
    initial_states=initial_states,
)
for indicator, the_list in simulated_values.items():
    print(f'{indicator}: {np.mean(the_list)} [{target_probabilities[indicator]}]')
