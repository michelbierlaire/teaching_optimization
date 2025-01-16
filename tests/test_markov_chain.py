import unittest
from enum import Enum
from random import choice

from teaching_optimization.metropolis_hastings.markov_chain import (
    State,
    MarkovChain,
    ParallelMarkovChain,
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


class SimpleMarkovState(State):
    """Simple state for a Markov Chain, that is a random walk among 4 states."""

    def __init__(self, the_state: MachineCondition) -> None:

        self.state: MachineCondition = the_state

    def indicators(self) -> dict[str, float]:
        """Relevant indicators associated with a state."""

        indicators = {condition.value: 0.0 for condition in MachineCondition}
        indicators[self.state.value] = 1.0
        return indicators


def next_state(the_state: State) -> State:
    """
    Returns the next state of the Markov Chain

    :return: here we implement a simple random walk
    """
    return SimpleMarkovState(the_state=get_random_machine_condition())


class TestMarkovChain(unittest.TestCase):

    def setUp(self):
        """Set up the initial states and MarkovChain for testing."""
        the_chains = [
            MarkovChain(
                initial_state=SimpleMarkovState(the_state=MachineCondition.STATE_A),
                transition_function=next_state,
            ),
            MarkovChain(
                initial_state=SimpleMarkovState(the_state=MachineCondition.STATE_B),
                transition_function=next_state,
            ),
            MarkovChain(
                initial_state=SimpleMarkovState(the_state=MachineCondition.STATE_C),
                transition_function=next_state,
            ),
            MarkovChain(
                initial_state=SimpleMarkovState(the_state=MachineCondition.STATE_D),
                transition_function=next_state,
            ),
        ]
        self.markov_chain = ParallelMarkovChain(
            the_chains=the_chains, stationarity_threshold=1.1
        )

    def test_initialization(self):
        """Test if the MarkovChain initializes correctly."""
        self.assertEqual(self.markov_chain.number_of_parallel_chains, 4)
        self.assertEqual(self.markov_chain.current_length, 1024)
        self.assertEqual(self.markov_chain.log2_current_length, 10)

    def test_current_length_property(self):
        """Test the current_length property."""
        self.assertEqual(self.markov_chain.current_length, 1 << 10)

    def test_extract_indicator_values(self):
        """Test extracting indicator values."""
        indicators = self.markov_chain.extract_indicator_values(
            MachineCondition.STATE_A.value
        )
        self.assertIsInstance(indicators, list)
        self.assertTrue(all(isinstance(val, float) for val in indicators))

    def test_number_of_parallel_chains(self):
        """Test the number_of_parallel_chains property."""
        self.assertEqual(self.markov_chain.number_of_parallel_chains, 4)

    def test_generate_one_sequence(self):
        """Test generating one sequence of states."""
        sequence = self.markov_chain._generate_one_sequence(0)
        self.assertEqual(len(sequence), self.markov_chain.current_length)
        self.assertTrue(all(isinstance(state, SimpleMarkovState) for state in sequence))

    def test_generate_sequences(self):
        """Test generating all sequences."""
        sequences = self.markov_chain._generate_sequences()
        self.assertEqual(len(sequences), 4)
        self.assertTrue(
            all(len(seq) == self.markov_chain.current_length for seq in sequences)
        )

    def test_double_chain_length(self):
        """Test doubling the chain length."""
        initial_length = self.markov_chain.current_length
        self.markov_chain._double_chain_length()
        self.assertEqual(self.markov_chain.current_length, 2 * initial_length)

    def test_generate_indicator_lists(self):
        """Test generating indicator lists."""
        indicator_lists = self.markov_chain._generate_indicator_lists()
        self.assertIsInstance(indicator_lists, dict)
        self.assertTrue(
            all(isinstance(key, str) for key in indicator_lists.keys())
        )  # Indicator names
        self.assertTrue(
            all(isinstance(value, list) for value in indicator_lists.values())
        )  # Indicator lists

    def test_variances_calculations(self):
        """Test the variance calculations."""
        variances = self.markov_chain._variances_calculations()
        self.assertIsInstance(variances, dict)
        self.assertTrue(all(isinstance(value, float) for value in variances.values()))

    def test_is_stationary(self):
        """Test if the chain detects stationarity."""
        stationary, message = self.markov_chain.is_stationary()
        self.assertIsInstance(stationary, bool)
        self.assertIsInstance(message, str)

    def test_reach_stationarity(self):
        """Test if the chain reaches stationarity."""
        reached_stationarity = self.markov_chain.reach_stationarity()
        self.assertIsInstance(reached_stationarity, bool)

    def test_invalid_sequence_id(self):
        """Test generating sequence with an invalid ID."""
        with self.assertRaises(ValueError):
            self.markov_chain._generate_one_sequence(10)

    def test_maximum_length_error(self):
        """Test error when the maximum chain length is exceeded."""
        self.markov_chain.log2_current_length = self.markov_chain.log2_maximum_length
        with self.assertRaises(ValueError):
            self.markov_chain._double_chain_length()


class TestSimpleMarkovState(unittest.TestCase):

    def test_indicators(self):
        """Test the indicators method."""
        state = SimpleMarkovState(MachineCondition.STATE_A)
        indicators = state.indicators()
        self.assertIsInstance(indicators, dict)
        self.assertEqual(indicators[MachineCondition.STATE_A.value], 1.0)
        for key, value in indicators.items():
            if key != MachineCondition.STATE_A.value:
                self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
