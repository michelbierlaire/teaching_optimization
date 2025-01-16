"""Implementation of the Metropolis-Hastings (MH) algorithm"""

from __future__ import annotations

import logging
from random import random
from typing import Protocol

import numpy as np
from teaching_optimization.metropolis_hastings.markov_chain import (
    State,
    MarkovChain,
    ParallelMarkovChain,
)

logger = logging.getLogger(__name__)


class ProposalState(Protocol):
    """Protocol of the function that defines the next state of the Markov Chain. It must return the logarithm
    of the forward and the backward transition probabilities"""

    def __call__(self, current_state: State) -> tuple[State, float, float]:
        """Generate the next state of the proposal Markov chain, together with the log of the forward and back
        transition probability

        :param current_state: current state
        :return: tuple containing the next state, the log of the forward transition probability, the log of the
        backward transition probability.
        """
        ...


class TargetProbability(Protocol):
    """Protocol of the function that provides the target unnormalized probability for the Metropolis-Hastings
    algorithm"""

    def __call__(self, the_state: State) -> float: ...


def metropolis_hastings_algorithm(
    proposal_chain: ProposalState,
    log_target_probability: TargetProbability,
    initial_states: list[State],
) -> dict[str, list[float]]:
    """Simulate the target distribution with the Metropolis Hastings algorithm.

    :param proposal_chain: function implementing the proposal chain for the MH algorithm
    :param log_target_probability: function returning the log of the unnormalized target probability for a given state
    :param initial_states: list of (ideally 4) initial states
    :return: a dict associating each indicator with a list of simulated values
    """

    number_of_candidates = 0
    accepted_candidates = 0

    def next_state(current_state: State) -> State:
        """
        Returns the next state of the Markov Chain

        :return:the next state
        """
        nonlocal number_of_candidates, accepted_candidates

        candidate, log_forward_probability, log_backward_probability = proposal_chain(
            current_state=current_state
        )
        number_of_candidates += 1
        current_target_log_probability = log_target_probability(the_state=current_state)
        candidate_target_log_probability = log_target_probability(the_state=candidate)
        log_alpha = min(
            candidate_target_log_probability
            + log_backward_probability
            - current_target_log_probability
            - log_forward_probability,
            0,
        )
        log_random_number = np.log(random())
        if log_random_number < log_alpha:
            accepted_candidates += 1
            return candidate
        return current_state

    the_chains = [
        MarkovChain(initial_state=the_state, transition_function=next_state)
        for the_state in initial_states
    ]
    the_parallel_chains = ParallelMarkovChain(
        the_chains=the_chains, stationarity_threshold=1.1
    )
    is_stationary = the_parallel_chains.reach_stationarity()
    print(f'Acceptance rate: {100 * accepted_candidates / number_of_candidates:.1f}%')
    return the_parallel_chains.extract_all_indicator_values()
