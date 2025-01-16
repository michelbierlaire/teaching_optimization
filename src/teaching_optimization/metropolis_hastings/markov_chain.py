"""Abstract class for the state representation

Michel Bierlaire
Fri Nov 29 10:06:26 2024
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from itertools import chain
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class State(ABC):
    """
    Abstract class representing the state of a Markov chain
    """

    @abstractmethod
    def indicators(self) -> dict[str, float]:
        """Relevant indicators associated with a state."""
        ...


class NextState(Protocol):
    """Protocol of the function that defined the next state of the Markov Chain"""

    def __call__(self, x: State) -> State: ...


class MarkovChain:
    """Implements a Markov Chain"""

    def __init__(
        self,
        initial_state: State,
        transition_function: NextState,
    ) -> None:
        """Constructor"""
        self.current_state = initial_state
        self.transition_function = transition_function

    def __iter__(self) -> State:
        """Make the Markov Chain iterable"""
        while True:  # Infinite generator for Markov chain
            yield self.current_state
            self.current_state = self.transition_function(self.current_state)

    def generate_sequence(self, length: int) -> list[State]:
        """Generate a chain, a sequence of states of given length"""
        return [state for _, state in zip(range(length), self)]


class ParallelMarkovChain:
    """Implements a Markov Chain"""

    def __init__(
        self,
        the_chains: list[MarkovChain],
        stationarity_threshold: float,
        log2_initial_length: int = 12,
        log2_maximum_length: int = 100,
    ) -> None:
        """Constructor

        :param the_chains: list containing each parallel chain
        :param stationarity_threshold: when the potential reduction has reached this threshold, we consider that the
            chain has reached stationarity (see Gelman et al. Section 11.5)
        :param log2_initial_length: log_2 of the length of the initial chain. It will be increased if insufficient.
        :param log2_maximum_length: log_2 of the maximum length of the chain.

        """
        self.the_chains = the_chains
        self.stationarity_threshold = stationarity_threshold
        self.log2_current_length: int = log2_initial_length
        self.log2_maximum_length: int = log2_maximum_length
        # Warmup
        logging.info(f'Warming up: {self.current_length} iterations')
        _ = self._generate_sequences()
        # Generate sequences
        self.sequences = self._generate_sequences()
        self.indicator_lists: dict[str, list[list[float]]] = (
            self._generate_indicator_lists()
        )

    @property
    def current_length(self) -> int:
        """Current length of the chain"""
        return 1 << self.log2_current_length

    def extract_all_indicator_values(self) -> dict[str, list[float]]:
        """Extract all simulate values"""
        return {
            the_indicator: self.extract_indicator_values(indicator_name=the_indicator)
            for the_indicator in self.indicator_lists
        }

    def extract_indicator_values(self, indicator_name: str) -> list[float]:
        """Extract the simulated values of one indicator

        :param indicator_name: name of the indicator
        :return: the list of values
        """
        the_nested_list = self.indicator_lists[indicator_name]
        flattened_list = list(chain.from_iterable(the_nested_list))
        return flattened_list

    @property
    def number_of_parallel_chains(self) -> int:
        """
        :return: number of chains running in parallel
        """
        return len(self.the_chains)

    def _generate_one_sequence(self, sequence_id) -> list[State]:
        """Generate one sequence of states of the chain

        :param sequence_id: index of the sequence to generate
        :return:  list of generated states
        """
        if sequence_id >= self.number_of_parallel_chains:
            raise ValueError(
                f'There are only {self.number_of_parallel_chains} concurrent chains. Chain id {sequence_id} is '
                f'invalid.'
            )
        return self.the_chains[sequence_id].generate_sequence(
            length=self.current_length
        )

    def _generate_sequences(self) -> list[list[State]]:
        """Before generating the sequence, we generate a sequence of the same length that is dropped to warm up the
        chain."""

        results = []
        with ThreadPoolExecutor(max_workers=self.number_of_parallel_chains) as executor:
            futures = {
                executor.submit(self._generate_one_sequence, chain_id): chain_id
                for chain_id in range(self.number_of_parallel_chains)
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _double_chain_length(self) -> None:
        """Generate parallel sequences twice as long as the previous one"""
        if self.log2_current_length >= self.log2_maximum_length:
            raise ValueError(
                f'The Markov chain has reached its maximum length {self.current_length}'
            )
        self.log2_current_length += 1
        logger.info(f'Generate sequences of length {2**self.log2_current_length}.')
        self.sequences = self._generate_sequences()
        self.indicator_lists = self._generate_indicator_lists()

    def _generate_indicator_lists(
        self,
    ) -> dict[str, list[list[float]]]:
        """
        If S is the number of parallel chains, generate M=2S lists of indicator values for each indicator.
        :return: A list of lists, each containing M lists of indicator values.
        """
        list_length = len(self.sequences[0]) // 2

        # Split the lists into two parts
        first_parts = [lst[:list_length] for lst in self.sequences]
        second_parts = [lst[list_length:] for lst in self.sequences]

        # Extract all indicator names from the first State object
        all_indicators = self.sequences[0][0].indicators().keys()

        # Generate a list of lists of dicts
        first_part_values = [
            [state.indicators() for state in part] for part in first_parts
        ]
        second_part_values = [
            [state.indicators() for state in part] for part in second_parts
        ]
        all_values = first_part_values + second_part_values
        result_dict = {indicator: [] for indicator in all_indicators}

        # Reorganize the data into a dict of lists of lists
        for indicator in all_indicators:
            result_dict[indicator] = [
                [state_dict[indicator] for state_dict in part] for part in all_values
            ]
        return result_dict

    def _variances_calculations(self) -> dict[str, float]:
        """Calculates the potential scale reduction Eq (11.4) in Gelman et al."""

        data = self._generate_indicator_lists()
        results = {}
        for indicator, sequences in data.items():
            number_of_sequences = len(sequences)  # Number of sequences
            length_of_sequences = len(sequences[0])  # Length of each sequence

            # Mean of each sequence (bar_theta_m)
            sequence_means = [np.mean(seq) for seq in sequences]

            # Mean of the means (bar_theta)
            mean_of_means = np.mean(sequence_means)

            # Between-sequence variance (B)
            between_sequence_variance = (
                length_of_sequences / (number_of_sequences - 1)
            ) * sum((mean - mean_of_means) ** 2 for mean in sequence_means)

            # Within-sequence variance (W)
            within_variances = [
                (1 / (length_of_sequences - 1))
                * sum((val - seq_mean) ** 2 for val in seq)
                for seq, seq_mean in zip(sequences, sequence_means)
            ]
            within_sequence_variance = float(np.mean(within_variances))

            # Store results for this parameter
            potential_reduction = np.sqrt(
                between_sequence_variance
                / (length_of_sequences * within_sequence_variance)
                + ((length_of_sequences - 1) / length_of_sequences)
            )
            logger.info(
                f'{indicator}: {between_sequence_variance=}, {within_sequence_variance=}, {potential_reduction=}'
            )
            results[indicator] = potential_reduction
        return results

    def is_stationary(self) -> tuple[bool, str]:
        """Check if the chain(s) have reached stationarity

        :return: a tuple with a boolean, and a string for the description of the diagnostic.
        """
        potential_reductions = self._variances_calculations()
        stationary: bool = True
        diagnostics: list[str] = []
        for indicator, reduction in potential_reductions.items():
            if reduction > self.stationarity_threshold:
                stationary = False
                diagnostics.append(
                    f'{indicator} [R_n = {reduction} > {self.stationarity_threshold}]'
                )
                diagnostics.append(
                    f'{indicator} [R_n = {reduction} > {self.stationarity_threshold}]'
                )
        if stationary:
            message = f'The potential reduction is below the threshold {self.stationarity_threshold} for all indicators'
            logger.info(message)
            return True, message
        message = (
            f'The potential reduction is above the threshold {self.stationarity_threshold} '
            f'for the following indicators: ' + ', '.join(diagnostics)
        )
        logger.info(message)
        return False, message

    def reach_stationarity(self) -> bool:
        """Run the Markov chain until it reaches stationarity

        :return: if True, the chain has reached stationarity for all indicators.
            If False, it has reached the maximum length.
        """
        while self.log2_current_length < self.log2_maximum_length:
            if self.is_stationary():
                return True
            self._double_chain_length()
        return False
