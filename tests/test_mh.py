import unittest
from enum import Enum
from random import choice, seed

import numpy as np
from teaching_optimization.metropolis_hastings.markov_chain import State
from teaching_optimization.metropolis_hastings.metropolis_hastings import (
    MetropolisHastingsState,
)


class MachineCondition(Enum):
    """States of the machine (example from the lecture)"""

    STATE_A = 'Perfect condition'
    STATE_B = 'Partially damaged'
    STATE_C = 'Seriously damaged'
    STATE_D = 'Completely useless'


def get_random_machine_condition() -> MachineCondition:
    """Select one randon state"""
    return choice(list(MachineCondition))


class SimpleMarkovState(MetropolisHastingsState):
    """Simple state for a Markov Chain, that is a random walk among 4 states."""

    target_probabilities = {
        'Perfect condition': 5 / 8,
        'Partially damaged': 1 / 4,
        'Seriously damaged': 3 / 32,
        'Completely useless': 1 / 32,
    }

    def __init__(self, the_state: MachineCondition) -> None:

        self.state: MachineCondition = the_state

    def next_state(self) -> tuple[State, float, float]:
        """
        Returns the next state of the Markov Chain

        :return: tuple with three elements: the next state, the log of the forward transition probability,
        the log of the backward transition probability
        """

        return SimpleMarkovState(the_state=get_random_machine_condition()), 0.25, 0.25

    def indicators(self) -> dict[str, float]:
        """Relevant indicators associated with a state."""

        indicators = {condition.value: 0.0 for condition in MachineCondition}
        indicators[self.state.value] = 1.0
        return indicators

    def log_target_probability(self) -> float:
        """Returns the logarithm of the target probability for the stationary distribution"""
        return np.log(self.target_probabilities[self.state.value])


class TestMetropolisHastings(unittest.TestCase):
    def setUp(self):
        """Set up the initial state for testing."""
        seed(42)  # For reproducibility
        self.initial_state = SimpleMarkovState(MachineCondition.STATE_A)

    def test_initial_state(self):
        """Test that the initial state is correctly set."""
        self.assertEqual(self.initial_state.state, MachineCondition.STATE_A)

    def test_log_target_probability(self):
        """Test that the log target probability is computed correctly."""
        log_prob = self.initial_state.log_target_probability()
        expected_log_prob = np.log(
            SimpleMarkovState.target_probabilities['Perfect condition']
        )
        self.assertAlmostEqual(log_prob, expected_log_prob, places=6)

    def test_indicators(self):
        """Test that the indicators are correctly returned."""
        indicators = self.initial_state.indicators()
        expected_indicators = {
            'Perfect condition': 1.0,
            'Partially damaged': 0.0,
            'Seriously damaged': 0.0,
            'Completely useless': 0.0,
        }
        self.assertDictEqual(indicators, expected_indicators)

    def test_next_state_transition(self):
        """Test the next_state method for state transitions."""
        next_state, log_forward, log_backward = self.initial_state.next_state()
        self.assertIsInstance(next_state, SimpleMarkovState)
        self.assertIn(next_state.state, MachineCondition)
        self.assertAlmostEqual(log_forward, 0.25, places=6)
        self.assertAlmostEqual(log_backward, 0.25, places=6)


if __name__ == '__main__':
    unittest.main()
